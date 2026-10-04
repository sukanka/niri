use std::cmp::max;
use std::iter::{once, zip};
use std::rc::Rc;

use anyhow::{ensure, Context as _};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::gles::{ffi, link_program, GlesError, GlesRenderer, GlesTexture};
use smithay::backend::renderer::vulkan::{
    CustomPass, CustomUniform, CustomUniformValue, VulkanFrame, VulkanPixelProgram, VulkanRenderer,
    VulkanTexture,
};
use smithay::backend::renderer::{ContextId, Offscreen as _, Renderer as _, Texture as _};
use smithay::gpu_span_location;
use smithay::utils::{Buffer, Size};

use crate::render_helpers::shaders::Shaders;

#[derive(Debug)]
pub struct Blur {
    program: BlurProgram,
    /// Context ID of the renderer that created the program and the textures.
    renderer_context_id: ContextId<GlesTexture>,
    /// Output texture followed by intermediate textures, large to small.
    ///
    /// Created lazily and stored here to avoid recreating blur textures frequently.
    textures: Vec<GlesTexture>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct BlurOptions {
    pub passes: u8,
    pub offset: f64,
}

impl From<niri_config::Blur> for BlurOptions {
    fn from(config: niri_config::Blur) -> Self {
        Self {
            passes: config.passes,
            offset: config.offset,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlurProgram(Rc<BlurProgramInner>);

#[derive(Debug)]
struct BlurProgramInner {
    down: BlurProgramInternal,
    up: BlurProgramInternal,
}

#[derive(Debug)]
struct BlurProgramInternal {
    program: ffi::types::GLuint,
    uniform_tex: ffi::types::GLint,
    uniform_half_pixel: ffi::types::GLint,
    uniform_offset: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

unsafe fn compile_program(gl: &ffi::Gles2, src: &str) -> Result<BlurProgramInternal, GlesError> {
    let program = unsafe { link_program(gl, include_str!("shaders/blur.vert"), src)? };

    let vert = c"vert";
    let tex = c"tex";
    let half_pixel = c"half_pixel";
    let offset = c"offset";

    Ok(BlurProgramInternal {
        program,
        uniform_tex: gl.GetUniformLocation(program, tex.as_ptr()),
        uniform_half_pixel: gl.GetUniformLocation(program, half_pixel.as_ptr()),
        uniform_offset: gl.GetUniformLocation(program, offset.as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr()),
    })
}

impl BlurProgram {
    pub fn compile(renderer: &mut GlesRenderer) -> anyhow::Result<Self> {
        renderer
            .with_context(move |gl| unsafe {
                let down = compile_program(gl, include_str!("shaders/blur_down.frag"))
                    .context("error compiling blur_down shader")?;
                let up = compile_program(gl, include_str!("shaders/blur_up.frag"))
                    .context("error compiling blur_up shader")?;
                Ok(Self(Rc::new(BlurProgramInner { down, up })))
            })
            .context("error making GL context current")?
    }

    pub fn destroy(self, renderer: &mut GlesRenderer) -> Result<(), GlesError> {
        renderer.with_context(move |gl| unsafe {
            gl.DeleteProgram(self.0.down.program);
            gl.DeleteProgram(self.0.up.program);
        })
    }
}

impl Blur {
    pub fn new(renderer: &mut GlesRenderer) -> Option<Self> {
        let program = Shaders::get(renderer)?.blur.clone()?;
        Some(Self {
            program,
            renderer_context_id: renderer.context_id(),
            textures: Vec::new(),
        })
    }

    pub fn context_id(&self) -> ContextId<GlesTexture> {
        self.renderer_context_id.clone()
    }

    pub fn prepare_textures(
        &mut self,
        mut create_texture: impl FnMut(Fourcc, Size<i32, Buffer>) -> Result<GlesTexture, GlesError>,
        source: &GlesTexture,
        options: BlurOptions,
    ) -> anyhow::Result<()> {
        let _span = tracy_client::span!("Blur::prepare_textures");

        let passes = options.passes.clamp(1, 31) as usize;
        let size = source.size();

        if let Some(output) = self.textures.first_mut() {
            let old_size = output.size();
            if old_size != size {
                trace!(
                    "recreating textures: output size changed from {} × {} to {} × {}",
                    old_size.w,
                    old_size.h,
                    size.w,
                    size.h
                );
                self.textures.clear();
            } else if !output.is_unique_reference() {
                debug!("recreating textures: not unique",);
                // We only need to recreate the output texture here, but this case shouldn't really
                // happen anyway, and this is simpler.
                self.textures.clear();
            }
        }

        // Create any missing textures.
        let mut w = size.w;
        let mut h = size.h;
        for i in 0..=passes {
            let size = Size::new(w, h);
            w = max(1, w / 2);
            h = max(1, h / 2);

            if self.textures.len() > i {
                // This texture already exists.
                continue;
            }

            // debug!("creating texture for step {i} sized {w} × {h}");

            let texture: GlesTexture =
                create_texture(Fourcc::Abgr8888, size).context("error creating texture")?;
            self.textures.push(texture);
        }

        // Drop any no longer needed textures.
        self.textures.drain(passes + 1..);

        Ok(())
    }

    pub fn render(
        &mut self,
        renderer: &mut GlesRenderer,
        source: &GlesTexture,
        options: BlurOptions,
    ) -> anyhow::Result<GlesTexture> {
        let _span = tracy_client::span!("Blur::render");
        trace!("rendering blur");

        crate::audit_texture_program!("blur");

        ensure!(
            renderer.context_id() == self.renderer_context_id,
            "wrong renderer"
        );

        let passes = options.passes.clamp(1, 31) as usize;
        let size = source.size();

        ensure!(
            self.textures.len() == passes + 1,
            "wrong textures len: expected {}, got {}",
            passes + 1,
            self.textures.len()
        );

        let output = &mut self.textures[0];
        ensure!(
            output.size() == size,
            "wrong output texture size: expected {size:?}, got {:?}",
            output.size()
        );

        ensure!(
            output.is_unique_reference(),
            "output texture has a non-unique reference"
        );

        renderer.with_profiled_framebuffer(gpu_span_location!("Blur::render"), |gl| unsafe {
            while gl.GetError() != ffi::NO_ERROR {}

            gl.Disable(ffi::BLEND);
            gl.Disable(ffi::SCISSOR_TEST);

            gl.ActiveTexture(ffi::TEXTURE0);

            let program = &self.program.0.down;
            gl.UseProgram(program.program);
            gl.Uniform1i(program.uniform_tex, 0);
            gl.Uniform1f(program.uniform_offset, options.offset as f32);

            let vertices: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
            gl.EnableVertexAttribArray(program.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                program.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                vertices.as_ptr().cast(),
            );

            let src = once(source).chain(&self.textures[1..]);
            let dst = &self.textures[1..];
            for (src, dst) in zip(src, dst) {
                let dst_size = dst.size();
                let w = dst_size.w;
                let h = dst_size.h;
                gl.Viewport(0, 0, w, h);

                // During downsampling, half_pixel is half of the destination pixel.
                gl.Uniform2f(program.uniform_half_pixel, 0.5 / w as f32, 0.5 / h as f32);

                let src = src.tex_id();
                let dst = dst.tex_id();

                trace!("drawing down {src} to {dst}");
                gl.FramebufferTexture2D(
                    ffi::FRAMEBUFFER,
                    ffi::COLOR_ATTACHMENT0,
                    ffi::TEXTURE_2D,
                    dst,
                    0,
                );

                gl.BindTexture(ffi::TEXTURE_2D, src);
                gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
                gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
                gl.TexParameteri(
                    ffi::TEXTURE_2D,
                    ffi::TEXTURE_WRAP_S,
                    ffi::CLAMP_TO_EDGE as i32,
                );
                gl.TexParameteri(
                    ffi::TEXTURE_2D,
                    ffi::TEXTURE_WRAP_T,
                    ffi::CLAMP_TO_EDGE as i32,
                );

                gl.DrawArrays(ffi::TRIANGLES, 0, 6);
            }

            gl.DisableVertexAttribArray(program.attrib_vert as u32);

            // Up
            let program = &self.program.0.up;
            gl.UseProgram(program.program);
            gl.Uniform1i(program.uniform_tex, 0);
            gl.Uniform1f(program.uniform_offset, options.offset as f32);

            let vertices: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
            gl.EnableVertexAttribArray(program.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                program.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                vertices.as_ptr().cast(),
            );

            let src = self.textures.iter().rev();
            let dst = self.textures.iter().rev().skip(1);
            for (src, dst) in zip(src, dst) {
                let dst_size = dst.size();
                let w = dst_size.w;
                let h = dst_size.h;
                gl.Viewport(0, 0, w, h);

                // During upsampling, half_pixel is half of the source pixel.
                let src_size = src.size();
                let src_w = src_size.w as f32;
                let src_h = src_size.h as f32;
                gl.Uniform2f(program.uniform_half_pixel, 0.5 / src_w, 0.5 / src_h);

                let src = src.tex_id();
                let dst = dst.tex_id();

                trace!("drawing up {src} to {dst}");
                gl.FramebufferTexture2D(
                    ffi::FRAMEBUFFER,
                    ffi::COLOR_ATTACHMENT0,
                    ffi::TEXTURE_2D,
                    dst,
                    0,
                );

                gl.BindTexture(ffi::TEXTURE_2D, src);
                gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
                gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
                gl.TexParameteri(
                    ffi::TEXTURE_2D,
                    ffi::TEXTURE_WRAP_S,
                    ffi::CLAMP_TO_EDGE as i32,
                );
                gl.TexParameteri(
                    ffi::TEXTURE_2D,
                    ffi::TEXTURE_WRAP_T,
                    ffi::CLAMP_TO_EDGE as i32,
                );

                gl.DrawArrays(ffi::TRIANGLES, 0, 6);
            }

            gl.DisableVertexAttribArray(program.attrib_vert as u32);
        })?;

        Ok(self.textures[0].clone())
    }
}

/// Compiled Vulkan dual-Kawase blur programs.
#[derive(Debug, Clone)]
pub struct VulkanBlurProgram {
    pub down: VulkanPixelProgram,
    pub up: VulkanPixelProgram,
}

/// Vulkan variant of [`Blur`], rendering the ping-pong passes into the frame's command
/// buffer via [`VulkanFrame::render_custom_passes`].
#[derive(Debug)]
pub struct VulkanBlur {
    renderer_context_id: ContextId<VulkanTexture>,
    program: VulkanBlurProgram,
    /// Output texture followed by intermediate textures, large to small.
    textures: Vec<VulkanTexture>,
}

impl VulkanBlur {
    pub fn new(renderer: &mut VulkanRenderer) -> Option<Self> {
        let program = renderer.user_data().get::<Shaders>()?.blur_vulkan.clone()?;
        Some(Self {
            renderer_context_id: renderer.context_id(),
            program,
            textures: Vec::new(),
        })
    }

    pub fn context_id(&self) -> ContextId<VulkanTexture> {
        self.renderer_context_id.clone()
    }

    pub fn prepare_textures(
        &mut self,
        renderer: &mut VulkanRenderer,
        source: &VulkanTexture,
        options: BlurOptions,
    ) -> anyhow::Result<()> {
        let _span = tracy_client::span!("VulkanBlur::prepare_textures");

        let passes = options.passes.clamp(1, 31) as usize;
        let size = source.size();

        if let Some(output) = self.textures.first_mut() {
            let old_size = output.size();
            if old_size != size {
                trace!(
                    "recreating textures: output size changed from {} × {} to {} × {}",
                    old_size.w,
                    old_size.h,
                    size.w,
                    size.h
                );
                self.textures.clear();
            } else if !output.is_unique_reference() {
                debug!("recreating textures: not unique");
                self.textures.clear();
            }
        }

        // Create any missing textures.
        let mut w = size.w;
        let mut h = size.h;
        for i in 0..=passes {
            let size = Size::new(w, h);
            w = max(1, w / 2);
            h = max(1, h / 2);

            if self.textures.len() > i {
                continue;
            }

            let texture: VulkanTexture = renderer
                .create_buffer(Fourcc::Abgr8888, size)
                .context("error creating texture")?;
            self.textures.push(texture);
        }

        // Drop any no longer needed textures.
        self.textures.drain(passes + 1..);

        Ok(())
    }

    pub fn render(
        &mut self,
        frame: &mut VulkanFrame<'_, '_>,
        source: &VulkanTexture,
        options: BlurOptions,
    ) -> anyhow::Result<VulkanTexture> {
        let _span = tracy_client::span!("VulkanBlur::render");
        trace!("rendering vulkan blur");

        crate::audit_texture_program!("blur");

        let passes = options.passes.clamp(1, 31) as usize;
        let size = source.size();

        ensure!(
            self.textures.len() == passes + 1,
            "wrong textures len: expected {}, got {}",
            passes + 1,
            self.textures.len()
        );

        let output = &self.textures[0];
        ensure!(
            output.size() == size,
            "wrong output texture size: expected {size:?}, got {:?}",
            output.size()
        );

        let offset = options.offset as f32;

        // Per-pass uniform storage must outlive the pass descriptors.
        let mut uniforms = Vec::with_capacity(passes * 2);
        // Down: during downsampling, half_pixel is half of the destination pixel.
        for dst in &self.textures[1..] {
            let dst_size = dst.size();
            uniforms.push([
                CustomUniform {
                    name: "half_pixel",
                    value: CustomUniformValue::Vec2([
                        0.5 / dst_size.w as f32,
                        0.5 / dst_size.h as f32,
                    ]),
                },
                CustomUniform {
                    name: "offset",
                    value: CustomUniformValue::Float(offset),
                },
            ]);
        }
        // Up: during upsampling, half_pixel is half of the source pixel.
        for src in self.textures.iter().rev().take(passes) {
            let src_size = src.size();
            uniforms.push([
                CustomUniform {
                    name: "half_pixel",
                    value: CustomUniformValue::Vec2([
                        0.5 / src_size.w as f32,
                        0.5 / src_size.h as f32,
                    ]),
                },
                CustomUniform {
                    name: "offset",
                    value: CustomUniformValue::Float(offset),
                },
            ]);
        }

        let mut textures = Vec::with_capacity(passes * 2);
        let down_src = once(source).chain(&self.textures[1..]);
        let down_dst = &self.textures[1..];
        for (src, dst) in zip(down_src, down_dst) {
            textures.push((dst, src));
        }
        let up_src = self.textures.iter().rev();
        let up_dst = self.textures.iter().rev().skip(1);
        for (src, dst) in zip(up_src, up_dst) {
            textures.push((dst, src));
        }

        let tex_bindings: Vec<[(&str, &VulkanTexture); 1]> =
            textures.iter().map(|(_, src)| [("tex", *src)]).collect();

        let mut pass_descs = Vec::with_capacity(textures.len());
        for (i, (dst, _)) in textures.iter().enumerate() {
            let program = if i < passes {
                &self.program.down
            } else {
                &self.program.up
            };
            pass_descs.push(CustomPass {
                dst,
                program,
                uniforms: &uniforms[i],
                textures: &tex_bindings[i],
            });
        }

        frame
            .render_custom_passes(&pass_descs)
            .context("error rendering blur passes")?;

        Ok(self.textures[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use smithay::backend::egl::native::EGLSurfacelessDisplay;
    use smithay::backend::egl::{EGLContext, EGLDisplay};
    use smithay::backend::renderer::{ExportMem as _, ImportMem as _};
    use smithay::utils::Rectangle;

    use super::*;

    fn renderer() -> Option<GlesRenderer> {
        let result = (|| -> anyhow::Result<_> {
            let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay)? };
            let context = EGLContext::new(&display)?;
            Ok(unsafe { GlesRenderer::new(context)? })
        })();
        match result {
            Ok(renderer) => Some(renderer),
            Err(err) => {
                if std::env::var_os("NIRI_TEST_REQUIRE_GPU")
                    .is_some_and(|v| !v.is_empty() && v != "0")
                {
                    panic!("blur test requires GLES: {err:#}");
                }
                eprintln!("skipping blur test: {err:#}");
                None
            }
        }
    }

    fn blur(renderer: &mut GlesRenderer) -> Blur {
        Blur {
            program: BlurProgram::compile(renderer).unwrap(),
            renderer_context_id: renderer.context_id(),
            textures: Vec::new(),
        }
    }

    fn pixels(renderer: &mut GlesRenderer, texture: &GlesTexture) -> Vec<u8> {
        let mapping = renderer
            .copy_texture(
                texture,
                Rectangle::from_size(texture.size()),
                Fourcc::Abgr8888,
            )
            .unwrap();
        let result = renderer.map_texture(&mapping).unwrap().to_vec();
        renderer
            .with_context(|gl| assert_eq!(unsafe { gl.GetError() }, ffi::NO_ERROR))
            .unwrap();
        result
    }

    #[test]
    fn repeated_blur_preserves_pixels_and_reuses_textures_after_readback() {
        let Some(mut renderer) = renderer() else {
            return;
        };
        let mut cached = blur(&mut renderer);
        for (size, passes) in [((16, 12), 3), ((10, 8), 2), ((16, 12), 3)] {
            let options = BlurOptions { passes, offset: 1. };
            let data: Vec<_> = (0..size.0 * size.1)
                .flat_map(|index| [(index * 19) as u8, (index * 7) as u8, 128, 255])
                .collect();
            let source = renderer
                .import_memory(&data, Fourcc::Abgr8888, size.into(), false)
                .unwrap();
            let mut reference = blur(&mut renderer);
            reference
                .prepare_textures(
                    |format, size| renderer.create_buffer(format, size),
                    &source,
                    options,
                )
                .unwrap();
            let reference_output = reference.render(&mut renderer, &source, options).unwrap();
            let expected = pixels(&mut renderer, &reference_output);
            drop(reference_output);
            reference.program.destroy(&mut renderer).unwrap();

            cached
                .prepare_textures(
                    |format, size| renderer.create_buffer(format, size),
                    &source,
                    options,
                )
                .unwrap();
            let ids: Vec<_> = cached.textures.iter().map(GlesTexture::tex_id).collect();
            for _ in 0..8 {
                cached
                    .prepare_textures(
                        |format, size| renderer.create_buffer(format, size),
                        &source,
                        options,
                    )
                    .unwrap();
                let output = cached.render(&mut renderer, &source, options).unwrap();
                assert_eq!(pixels(&mut renderer, &output), expected);
                drop(output);
                assert_eq!(
                    cached
                        .textures
                        .iter()
                        .map(GlesTexture::tex_id)
                        .collect::<Vec<_>>(),
                    ids
                );
            }
        }
        cached.program.destroy(&mut renderer).unwrap();
    }
}
