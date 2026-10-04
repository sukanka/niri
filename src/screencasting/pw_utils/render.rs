use std::collections::{HashMap, HashSet};

use anyhow::{ensure, Context as _};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{Buffer as _, Fourcc};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::RenderElement;
use smithay::backend::renderer::sync::SyncPoint;
use smithay::backend::renderer::{Color32F, ContextId, ErasedContextId};
use smithay::utils::{Physical, Scale, Size, Transform};

use super::{Shmbuf, SHM_BYTES_PER_PIXEL};
use crate::backend::tty_renderer::TtyOffscreen;
use crate::render_helpers::blend::set_sdr_capture_blend;
use crate::render_helpers::renderer::NiriCaptureRenderer;
use crate::render_helpers::{copy_framebuffer, create_texture};

#[derive(Debug, PartialEq)]
struct Parameters {
    context: ErasedContextId,
    size: Size<i32, Physical>,
    scale: Scale<f64>,
    reference_luminance: f64,
}

#[derive(Debug)]
struct ShmFrame {
    texture: TtyOffscreen,
    format: Fourcc,
    damage: OutputDamageTracker,
    /// SHM buffers which already contain the current image, including for cursor-only frames.
    copied: HashSet<i64>,
}

/// Damage belongs to the actual destination buffer, independently of the tracker used to
/// decide whether to dequeue a frame. Polling under backpressure cannot consume this damage.
#[derive(Debug, Default)]
pub(super) struct RenderCache {
    parameters: Option<Parameters>,
    dmabufs: HashMap<i64, OutputDamageTracker>,
    // Share one offscreen texture across SHM buffers, instead of retaining a full image per slot.
    shm: Option<ShmFrame>,
}

impl RenderCache {
    pub(super) fn prepare<R: NiriCaptureRenderer>(
        &mut self,
        renderer: &R,
        size: Size<i32, Physical>,
        scale: Scale<f64>,
        reference_luminance: f64,
    ) -> bool {
        let parameters = Parameters {
            context: ContextId::erased(&renderer.context_id()),
            size,
            scale,
            reference_luminance,
        };
        if self.parameters.as_ref() == Some(&parameters) {
            return false;
        }
        *self = Self {
            parameters: Some(parameters),
            ..Self::default()
        };
        true
    }

    pub(super) fn remove_buffer(&mut self, fd: i64) {
        self.dmabufs.remove(&fd);
        if let Some(shm) = &mut self.shm {
            shm.copied.remove(&fd);
        }
    }

    pub(super) fn render_dmabuf<R: NiriCaptureRenderer>(
        &mut self,
        renderer: &mut R,
        fd: i64,
        mut dmabuf: Dmabuf,
        elements: &[impl RenderElement<R>],
    ) -> anyhow::Result<SyncPoint> {
        let _span = tracy_client::span!("RenderCache::render_dmabuf");
        let size = self.parameters.as_ref().unwrap().size;
        ensure!(
            dmabuf.width() == size.w as u32 && dmabuf.height() == size.h as u32,
            "invalid capture buffer size"
        );
        let mut target = renderer.bind(&mut dmabuf).context("error binding dmabuf")?;
        self.render_dmabuf_target(renderer, fd, &mut target, elements)
    }

    fn render_dmabuf_target<R: NiriCaptureRenderer>(
        &mut self,
        renderer: &mut R,
        fd: i64,
        target: &mut R::Framebuffer<'_>,
        elements: &[impl RenderElement<R>],
    ) -> anyhow::Result<SyncPoint> {
        let params = self.parameters.as_ref().unwrap();
        set_sdr_capture_blend(renderer, params.reference_luminance);
        let damage = self.dmabufs.entry(fd).or_insert_with(|| {
            OutputDamageTracker::new(params.size, params.scale, Transform::Normal)
        });
        let result = (|| {
            let result = damage
                .render_output(renderer, target, 1, elements, Color32F::TRANSPARENT)
                .context("error rendering to dmabuf")?;
            Ok(result.sync)
        })();
        if result.is_err() {
            self.dmabufs.remove(&fd);
        }
        result
    }

    pub(super) fn render_shm<R: NiriCaptureRenderer>(
        &mut self,
        renderer: &mut R,
        fd: i64,
        buffer: &Shmbuf,
        fourcc: Fourcc,
        elements: &[impl RenderElement<R>],
    ) -> anyhow::Result<()> {
        let _span = tracy_client::span!("RenderCache::render_shm");
        let params = self.parameters.as_ref().unwrap();
        set_sdr_capture_blend(renderer, params.reference_luminance);
        let expected_size = params.size.w as usize * params.size.h as usize * SHM_BYTES_PER_PIXEL;
        ensure!(
            buffer.layout.size_usize() == expected_size,
            "invalid buffer size"
        );

        // Return the cache only on success: a failed draw or readback must retry with full damage.
        let mut shm = match self.shm.take().filter(|shm| shm.format == fourcc) {
            Some(shm) => shm,
            None => ShmFrame {
                texture: R::wrap_offscreen(create_texture(renderer, params.size, fourcc)?),
                format: fourcc,
                damage: OutputDamageTracker::new(params.size, params.scale, Transform::Normal),
                copied: HashSet::new(),
            },
        };
        {
            let texture =
                R::unwrap_offscreen(&mut shm.texture).context("wrong capture renderer")?;
            let mut target = renderer
                .bind(texture)
                .context("error binding capture texture")?;
            let result = shm
                .damage
                .render_output(renderer, &mut target, 1, elements, Color32F::TRANSPARENT)
                .context("error rendering capture texture")?;
            if result.damage.is_some() {
                shm.copied.clear();
            }
            if !shm.copied.contains(&fd) {
                let mapping = copy_framebuffer(renderer, &target, fourcc)?;
                let bytes = renderer
                    .map_texture(&mapping)
                    .context("error mapping capture texture")?;
                ensure!(
                    bytes.len() >= expected_size,
                    "short capture texture mapping"
                );
                buffer.mapping.copy_frame(&bytes[..expected_size]);
                shm.copied.insert(fd);
            }
        }
        self.shm = Some(shm);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::slice;

    use smithay::backend::renderer::element::Kind;

    use super::*;
    use crate::render_helpers::solid_color::{SolidColorBuffer, SolidColorRenderElement};
    use crate::screencasting::pw_utils::allocate_shmbuf;
    use crate::tests::gpu;

    fn pixels(buffer: &Shmbuf) -> Vec<u8> {
        let mut bytes = vec![0; buffer.layout.size_usize()];
        File::from(buffer.fd.try_clone().unwrap())
            .read_exact_at(&mut bytes, 0)
            .unwrap();
        bytes
    }

    fn texture_id(cache: &RenderCache) -> String {
        match &cache.shm.as_ref().unwrap().texture {
            TtyOffscreen::Gles(texture) => format!("{:?}", texture.tex_id()),
            TtyOffscreen::Vulkan(texture) => format!("{:?}", texture.image()),
            _ => unreachable!(),
        }
    }

    fn shm_buffer_rotation<R: NiriCaptureRenderer>(renderer: &mut R) {
        let mut cache = RenderCache::default();
        let size = (2, 1).into();
        assert!(cache.prepare(renderer, size, 1.0.into(), 203.));
        let a = allocate_shmbuf((2, 1).into()).unwrap();
        let b = allocate_shmbuf((2, 1).into()).unwrap();
        let mut solid = SolidColorBuffer::new((1., 1.), [1., 0., 0., 1.]);
        let red = SolidColorRenderElement::from_buffer(&solid, (0., 0.), 1., Kind::Unspecified);
        cache
            .render_shm(renderer, 1, &a, Fourcc::Argb8888, slice::from_ref(&red))
            .unwrap();
        let initial_texture = texture_id(&cache);
        let red_pixels = [0, 0, 255, 255, 0, 0, 0, 0];
        assert_eq!(pixels(&a), red_pixels);
        cache
            .render_shm(renderer, 2, &b, Fourcc::Argb8888, slice::from_ref(&red))
            .unwrap();
        assert_eq!(pixels(&b), red_pixels);

        // No scene change: a cursor metadata frame can reuse either slot's complete image.
        assert!(!cache.prepare(renderer, size, 1.0.into(), 203.));
        cache
            .render_shm(renderer, 1, &a, Fourcc::Argb8888, &[red])
            .unwrap();
        assert_eq!(pixels(&a), red_pixels);
        assert_eq!(texture_id(&cache), initial_texture);

        // Only one pixel changes. A slot not yet returned by the consumer stays intact;
        // when it returns it must receive the latest contents, even after intervening frames.
        solid.set_color([0., 1., 0., 1.]);
        let green = SolidColorRenderElement::from_buffer(&solid, (0., 0.), 1., Kind::Unspecified);
        cache
            .render_shm(renderer, 1, &a, Fourcc::Argb8888, slice::from_ref(&green))
            .unwrap();
        let green_pixels = [0, 255, 0, 255, 0, 0, 0, 0];
        assert_eq!(pixels(&a), green_pixels);
        assert_eq!(pixels(&b), red_pixels);
        cache
            .render_shm(renderer, 2, &b, Fourcc::Argb8888, slice::from_ref(&green))
            .unwrap();
        assert_eq!(pixels(&b), green_pixels);
        assert_eq!(texture_id(&cache), initial_texture);

        // Removing a PipeWire buffer invalidates its contents even if the fd is reused.
        cache.remove_buffer(2);
        b.mapping.clear();
        cache
            .render_shm(renderer, 2, &b, Fourcc::Argb8888, &[green])
            .unwrap();
        assert_eq!(pixels(&b), green_pixels);

        assert!(cache.prepare(renderer, size, 1.0.into(), 250.));
        assert!(cache.shm.is_none(), "color changes need a fresh full frame");
        cache
            .render_shm(
                renderer,
                1,
                &a,
                Fourcc::Argb8888,
                &[] as &[SolidColorRenderElement],
            )
            .unwrap();
        assert_eq!(
            pixels(&a),
            [0; 8],
            "a removed window must not leave stale pixels"
        );
    }

    fn dmabuf_target_rotation<R: NiriCaptureRenderer>(renderer: &mut R) {
        let mut cache = RenderCache::default();
        cache.prepare(renderer, (2, 1).into(), 1.0.into(), 203.);
        // Offscreen targets exercise the same per-buffer damage path without requiring a
        // particular GBM device or DMA-BUF modifier to be available to this renderer.
        let mut a = create_texture(renderer, (2, 1).into(), Fourcc::Abgr8888).unwrap();
        let mut b = create_texture(renderer, (2, 1).into(), Fourcc::Abgr8888).unwrap();
        let mut solid = SolidColorBuffer::new((1., 1.), [1., 0., 0., 1.]);
        for (fd, color) in [
            (1, [1., 0., 0., 1.]),
            (2, [1., 0., 0., 1.]),
            (1, [0., 1., 0., 1.]),
            (2, [0., 1., 0., 1.]),
        ] {
            let texture = if fd == 1 { &mut a } else { &mut b };
            solid.set_color(color);
            let element =
                SolidColorRenderElement::from_buffer(&solid, (0., 0.), 1., Kind::Unspecified);
            let mut target = renderer.bind(texture).unwrap();
            let _sync = cache
                .render_dmabuf_target(renderer, fd, &mut target, &[element])
                .unwrap();
            let mapping = copy_framebuffer(renderer, &target, Fourcc::Abgr8888).unwrap();
            let expected = [
                (color[0] * 255.) as u8,
                (color[1] * 255.) as u8,
                0,
                255,
                0,
                0,
                0,
                0,
            ];
            assert_eq!(renderer.map_texture(&mapping).unwrap(), expected);
        }
        for (fd, texture) in [(1, &mut a), (2, &mut b)] {
            let mut target = renderer.bind(texture).unwrap();
            let _sync = cache
                .render_dmabuf_target(renderer, fd, &mut target, &[] as &[SolidColorRenderElement])
                .unwrap();
            let mapping = copy_framebuffer(renderer, &target, Fourcc::Abgr8888).unwrap();
            assert_eq!(renderer.map_texture(&mapping).unwrap(), [0; 8]);
        }
    }

    #[test]
    fn shm_capture_rotation_gles() {
        if let Some(mut renderer) = gpu::gles_renderer() {
            shm_buffer_rotation(&mut renderer);
        }
    }

    #[test]
    fn shm_capture_rotation_vulkan() {
        if let Some(mut renderer) = gpu::vulkan_renderer() {
            shm_buffer_rotation(&mut renderer);
        }
    }

    #[test]
    fn dmabuf_damage_rotation_gles() {
        if let Some(mut renderer) = gpu::gles_renderer() {
            dmabuf_target_rotation(&mut renderer);
        }
    }

    #[test]
    fn dmabuf_damage_rotation_vulkan() {
        if let Some(mut renderer) = gpu::vulkan_renderer() {
            dmabuf_target_rotation(&mut renderer);
        }
    }
}
