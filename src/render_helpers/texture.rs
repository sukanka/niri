use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context as _};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{GlesError, GlesFrame, GlesRenderer, GlesTexture};
use smithay::backend::renderer::utils::{CommitCounter, DamageSet, OpaqueRegions};
use smithay::backend::renderer::{
    ErasedContextId, Frame as _, FrameContext as _, ImportMem, Renderer, Texture,
};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};

use super::memory::MemoryBuffer;
use super::renderer::{NiriCaptureRenderer, NiriRenderer};
use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};
use crate::backend::tty_renderer::{GlesApi, TtyOffscreen, VulkanApi};

/// Smithay's texture buffer, but with fractional scale.
#[derive(Debug, Clone)]
pub struct TextureBuffer<T: Texture> {
    id: Id,
    commit_counter: CommitCounter,
    renderer_context_id: ErasedContextId,
    texture: T,
    portable: Option<Arc<PortablePixels>>,
    scale: Scale<f64>,
    transform: Transform,
    opaque_regions: Vec<Rectangle<i32, Buffer>>,
}

/// Render element for a [`TextureBuffer`].
#[derive(Debug, Clone)]
pub struct TextureRenderElement<T: Texture> {
    buffer: TextureBuffer<T>,
    location: Point<f64, Logical>,
    alpha: f32,
    src: Option<Rectangle<f64, Logical>>,
    size: Option<Size<f64, Logical>>,
    kind: Kind,
}

impl<T: Texture + 'static> TextureBuffer<T> {
    pub fn from_texture<R: Renderer>(
        renderer: &R,
        texture: T,
        scale: impl Into<Scale<f64>>,
        transform: Transform,
        opaque_regions: Vec<Rectangle<i32, Buffer>>,
    ) -> Self
    where
        R::TextureId: 'static,
    {
        TextureBuffer {
            id: Id::new(),
            commit_counter: CommitCounter::default(),
            renderer_context_id: renderer.context_id().map::<TtyOffscreen>().erased(),
            texture,
            portable: None,
            scale: scale.into(),
            transform,
            opaque_regions,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_memory<R: NiriRenderer<NiriTextureId = T>>(
        renderer: &mut R,
        data: &[u8],
        format: Fourcc,
        size: impl Into<Size<i32, Buffer>>,
        flipped: bool,
        scale: impl Into<Scale<f64>>,
        transform: Transform,
        opaque_regions: Vec<Rectangle<i32, Buffer>>,
    ) -> Result<Self, R::Error> {
        let size = size.into();
        let texture = renderer.import_memory(data, format, size, flipped)?;
        let mut buffer =
            TextureBuffer::from_texture(renderer, texture, scale, transform, opaque_regions);
        if needs_portable_textures(renderer) {
            buffer.portable = Some(Arc::new(PortablePixels {
                data: data.into(),
                format,
                size,
                flipped,
                textures: Mutex::new(HashMap::new()),
            }));
        }
        Ok(buffer)
    }

    pub fn from_memory_buffer<R: NiriRenderer<NiriTextureId = T>>(
        renderer: &mut R,
        buffer: &MemoryBuffer,
    ) -> Result<Self, R::Error> {
        Self::from_memory(
            renderer,
            buffer.data(),
            buffer.format(),
            buffer.size(),
            false,
            buffer.scale(),
            buffer.transform(),
            Vec::new(),
        )
    }

    pub fn texture(&self) -> &T {
        &self.texture
    }

    /// Converts the texture type, keeping all buffer metadata.
    pub fn map_texture<U: Texture>(self, f: impl FnOnce(T) -> U) -> TextureBuffer<U> {
        TextureBuffer {
            id: self.id,
            commit_counter: self.commit_counter,
            renderer_context_id: self.renderer_context_id,
            texture: f(self.texture),
            portable: self.portable,
            scale: self.scale,
            transform: self.transform,
            opaque_regions: self.opaque_regions,
        }
    }

    pub fn texture_scale(&self) -> Scale<f64> {
        self.scale
    }

    pub fn set_texture_scale(&mut self, scale: impl Into<Scale<f64>>) {
        self.scale = scale.into();
    }

    pub fn texture_transform(&self) -> Transform {
        self.transform
    }

    pub fn set_texture_transform(&mut self, transform: Transform) {
        self.transform = transform;
    }
}

impl<T: Texture> TextureBuffer<T> {
    pub fn logical_size(&self) -> Size<f64, Logical> {
        self.texture
            .size()
            .to_f64()
            .to_logical(self.scale, self.transform)
    }
}

impl TextureBuffer<GlesTexture> {
    pub fn is_texture_reference_unique(&mut self) -> bool {
        self.texture.is_unique_reference()
    }
}

impl<T: Texture> TextureRenderElement<T> {
    pub fn from_texture_buffer(
        buffer: TextureBuffer<T>,
        location: impl Into<Point<f64, Logical>>,
        alpha: f32,
        src: Option<Rectangle<f64, Logical>>,
        size: Option<Size<f64, Logical>>,
        kind: Kind,
    ) -> Self {
        TextureRenderElement {
            buffer,
            location: location.into(),
            alpha,
            src,
            size,
            kind,
        }
    }

    pub fn buffer(&self) -> &TextureBuffer<T> {
        &self.buffer
    }
}

impl<T: Texture> TextureRenderElement<T> {
    pub fn logical_size(&self) -> Size<f64, Logical> {
        self.size
            .or_else(|| self.src.map(|src| src.size))
            .unwrap_or_else(|| self.buffer.logical_size())
    }

    pub fn logical_src(&self) -> Rectangle<f64, Logical> {
        self.src
            .unwrap_or_else(|| Rectangle::from_size(self.logical_size()))
    }
}

impl<T: Texture> Element for TextureRenderElement<T> {
    fn id(&self) -> &Id {
        &self.buffer.id
    }

    fn current_commit(&self) -> CommitCounter {
        self.buffer.commit_counter
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        let logical_geo = Rectangle::new(self.location, self.logical_size());
        logical_geo.to_physical_precise_round(scale)
    }

    fn transform(&self) -> Transform {
        self.buffer.transform
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.src
            .map(|src| {
                src.to_buffer(
                    self.buffer.scale,
                    self.buffer.transform,
                    &self.buffer.logical_size(),
                )
            })
            .unwrap_or_else(|| Rectangle::from_size(self.buffer.texture.size()).to_f64())
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        let texture_size = self.buffer.texture.size().to_f64();
        let src = self.src();

        self.buffer
            .opaque_regions
            .iter()
            .filter_map(|region| {
                let mut region = region.to_f64().intersection(src)?;

                region.loc -= src.loc;
                region = region.upscale(texture_size / src.size);

                let logical =
                    region.to_logical(self.buffer.scale, self.buffer.transform, &src.size);
                Some(logical.to_physical_precise_down(scale))
            })
            .collect()
    }

    fn alpha(&self) -> f32 {
        self.alpha
    }

    fn kind(&self) -> Kind {
        self.kind
    }
}

impl<R, T> RenderElement<R> for TextureRenderElement<T>
where
    R: Renderer<TextureId = T>,
    T: Texture + 'static,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        if frame.context_id().map::<TtyOffscreen>().erased() != self.buffer.renderer_context_id {
            warn!("trying to render texture from different renderer");
            return Ok(());
        }

        frame.render_texture_from_to(
            &self.buffer.texture,
            src,
            dest,
            damage,
            opaque_regions,
            self.buffer.transform,
            self.alpha,
        )
    }

    fn underlying_storage(&self, _renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

/// Render element for a [`TextureBuffer`] of the universal [`TtyOffscreen`] texture enum.
///
/// Draws into both the winit `GlesRenderer` and either TTY renderer variant. Frozen
/// textures from another render context are imported from their portable backing and
/// cached for that context. A missing backing reports an error instead of drawing an
/// unrelated native texture handle.
#[derive(Debug, Clone)]
pub struct UniversalTextureRenderElement(pub TextureRenderElement<TtyOffscreen>);

impl Element for UniversalTextureRenderElement {
    fn id(&self) -> &Id {
        self.0.id()
    }

    fn current_commit(&self) -> CommitCounter {
        self.0.current_commit()
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.0.geometry(scale)
    }

    fn transform(&self) -> Transform {
        self.0.transform()
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.0.src()
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        self.0.damage_since(scale, commit)
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        self.0.opaque_regions(scale)
    }

    fn alpha(&self) -> f32 {
        self.0.alpha()
    }

    fn kind(&self) -> Kind {
        self.0.kind()
    }
}

impl RenderElement<GlesRenderer> for UniversalTextureRenderElement {
    fn draw(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        let buffer = self.0.buffer();
        let context = frame.context_id().map::<TtyOffscreen>().erased();
        let texture = buffer
            .cached_texture(&context)
            .map(Ok)
            .unwrap_or_else(|| buffer.texture_for_renderer(frame.renderer().as_mut()))
            .map_err(|err| {
                warn!("error importing frozen texture: {err:#}");
                GlesError::MappingError
            })?;
        let TtyOffscreen::Gles(texture) = texture else {
            return Err(GlesError::MappingError);
        };

        frame.render_texture_from_to(
            &texture,
            src,
            dest,
            damage,
            opaque_regions,
            buffer.texture_transform(),
            self.0.alpha(),
            None,
            &[],
        )
    }

    fn underlying_storage(&self, _renderer: &mut GlesRenderer) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

impl<'render> RenderElement<TtyRenderer<'render>> for UniversalTextureRenderElement {
    fn draw(
        &self,
        frame: &mut TtyFrame<'render, '_, '_>,
        src: Rectangle<f64, Buffer>,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), TtyRendererError<'render>> {
        let buffer = self.0.buffer();
        let transform = buffer.texture_transform();
        let alpha = self.0.alpha();

        // MultiFrame must prepare, synchronize, and possibly reimport this texture before
        // its native handle can be sampled. Merely extracting MultiTexture::get skips that.
        if frame.context_id().map::<TtyOffscreen>().erased() == buffer.renderer_context_id {
            if let TtyOffscreen::Multi(texture) = buffer.texture() {
                return frame.render_texture_from_to(
                    texture,
                    src,
                    dest,
                    damage,
                    opaque_regions,
                    transform,
                    alpha,
                );
            }
        }

        match frame {
            TtyFrame::Gles(multi) => {
                let frame: &mut GlesFrame<'_, '_> = multi.as_mut();
                RenderElement::<GlesRenderer>::draw(
                    self,
                    frame,
                    src,
                    dest,
                    damage,
                    opaque_regions,
                    None,
                )
                .map_err(Into::into)
            }
            TtyFrame::Vulkan(multi) => {
                let frame: &mut smithay::backend::renderer::vulkan::VulkanFrame<'_, '_> =
                    multi.as_mut();
                let texture = buffer.texture_for_renderer(frame.renderer().as_mut()).map_err(|err| {
                    warn!("error importing frozen texture: {err:#}");
                    TtyRendererError::Vulkan(smithay::backend::renderer::multigpu::Error::Render(
                        smithay::backend::renderer::vulkan::VulkanError::ForeignTextureInPass,
                    ))
                })?;
                let TtyOffscreen::Vulkan(texture) = texture else {
                    unreachable!()
                };
                frame
                    .render_texture_from_to(
                        &texture,
                        src,
                        dest,
                        damage,
                        opaque_regions,
                        transform,
                        alpha,
                    )
                    .map_err(|err| {
                        TtyRendererError::Vulkan(
                            smithay::backend::renderer::multigpu::Error::Render(err),
                        )
                    })
            }
        }
    }

    fn underlying_storage(
        &self,
        _renderer: &mut TtyRenderer<'render>,
    ) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

/// A CPU backing store for immutable textures. Copies on another GPU are imported only once.
#[derive(Debug)]
struct PortablePixels {
    data: Arc<[u8]>,
    format: Fourcc,
    size: Size<i32, Buffer>,
    flipped: bool,
    textures: Mutex<HashMap<ErasedContextId, TtyOffscreen>>,
}

#[derive(Default)]
struct TexturePortability(Cell<bool>);

/// Enables backing up frozen GPU textures when outputs may use different render contexts.
pub fn set_texture_portability<R: NiriRenderer>(renderer: &mut R, enabled: bool) {
    let data = if let Some(renderer) = renderer.as_gles_renderer() {
        renderer.egl_context().user_data()
    } else if let Some(renderer) = renderer.as_vulkan_renderer() {
        renderer.user_data()
    } else {
        return;
    };
    data.insert_if_missing(TexturePortability::default);
    data.get::<TexturePortability>().unwrap().0.set(enabled);
}

pub fn needs_portable_textures<R: NiriRenderer>(renderer: &mut R) -> bool {
    if let Some(renderer) = renderer.as_gles_renderer() {
        renderer
            .egl_context()
            .user_data()
            .get::<TexturePortability>()
            .is_some_and(|flag| flag.0.get())
    } else if let Some(renderer) = renderer.as_vulkan_renderer() {
        renderer
            .user_data()
            .get::<TexturePortability>()
            .is_some_and(|flag| flag.0.get())
    } else {
        false
    }
}

impl TextureBuffer<TtyOffscreen> {
    // Keep native and already-imported draws on the frame fast path. Borrowing a GLES
    // frame's renderer also restores GL state on drop; avoid that work for every draw.
    fn cached_texture(&self, context: &ErasedContextId) -> Option<TtyOffscreen> {
        if *context == self.renderer_context_id && !matches!(self.texture, TtyOffscreen::Multi(_)) {
            return Some(self.texture.clone());
        }
        self.portable
            .as_ref()?
            .textures
            .lock()
            .unwrap()
            .get(context)
            .cloned()
    }

    /// Saves immutable contents while the source renderer is still available. Disabled for
    /// the traditional single-render-GPU mode. Normal live surfaces never use this path.
    pub fn make_portable<R: NiriRenderer>(&mut self, renderer: &mut R) -> anyhow::Result<()> {
        if self.portable.is_some() || !needs_portable_textures(renderer) {
            return Ok(());
        }
        // Surface MultiTextures need MultiFrame's synchronization and import preparation.
        // Their caller freezes them with render_to_texture before reaching this method.
        if matches!(self.texture, TtyOffscreen::Multi(_)) {
            bail!("surface snapshot must be frozen before preserving it for another GPU");
        }
        let texture = self.texture_for_renderer(renderer)?;
        let pixels = match texture {
            TtyOffscreen::Gles(texture) => {
                download_gles_pixels(renderer.as_gles_renderer().unwrap(), &texture)?
            }
            TtyOffscreen::Vulkan(texture) => download_pixels(
                renderer.as_vulkan_renderer().unwrap(),
                &texture,
                texture.is_y_inverted(),
            )?,
            TtyOffscreen::Multi(_) => unreachable!(),
        };
        self.portable = Some(Arc::new(pixels));
        Ok(())
    }

    /// Resolves a frozen texture for this renderer, preserving its source pixel format.
    pub fn texture_for_renderer<R: NiriRenderer>(
        &self,
        renderer: &mut R,
    ) -> anyhow::Result<TtyOffscreen> {
        let id = renderer.context_id().map::<TtyOffscreen>().erased();
        if id == self.renderer_context_id {
            return match &self.texture {
                TtyOffscreen::Multi(texture) => if let Some(renderer) = renderer.as_gles_renderer()
                {
                    texture
                        .get::<GlesApi>(&renderer.context_id())
                        .map(TtyOffscreen::Gles)
                } else if let Some(renderer) = renderer.as_vulkan_renderer() {
                    texture
                        .get::<VulkanApi>(&renderer.context_id())
                        .map(TtyOffscreen::Vulkan)
                } else {
                    None
                }
                .context("missing native snapshot texture"),
                texture => Ok(texture.clone()),
            };
        }
        let pixels = self
            .portable
            .as_ref()
            .context("frozen texture has no portable backing")?;
        let mut textures = pixels.textures.lock().unwrap();
        if let Some(texture) = textures.get(&id) {
            return Ok(texture.clone());
        }
        let texture = if let Some(renderer) = renderer.as_gles_renderer() {
            TtyOffscreen::Gles(renderer.import_memory(
                &pixels.data,
                pixels.format,
                pixels.size,
                pixels.flipped,
            )?)
        } else if let Some(renderer) = renderer.as_vulkan_renderer() {
            TtyOffscreen::Vulkan(renderer.import_memory(
                &pixels.data,
                pixels.format,
                pixels.size,
                pixels.flipped,
            )?)
        } else {
            bail!("unsupported renderer for a frozen texture");
        };
        textures.insert(id, texture.clone());
        Ok(texture)
    }
}

/// Smithay's GLES mapping currently exposes four bytes per pixel even for RGBA16F.
/// Read larger pixels into an owned, correctly sized buffer instead of truncating HDR data.
fn download_gles_pixels(
    renderer: &mut GlesRenderer,
    texture: &GlesTexture,
) -> anyhow::Result<PortablePixels> {
    use smithay::backend::renderer::gles::ffi;
    use smithay::backend::renderer::gles::format::{fourcc_to_gl_formats, gl_bpp};
    use smithay::backend::renderer::Bind as _;

    let format = texture
        .format()
        .context("snapshot texture has no pixel format")?;
    let (_, read_format, read_type) =
        fourcc_to_gl_formats(format).context("unsupported snapshot format")?;
    let bytes_per_pixel =
        gl_bpp(read_format, read_type).context("unknown snapshot pixel size")? / 8;
    if bytes_per_pixel <= 4 {
        return download_pixels(renderer, texture, texture.is_y_inverted());
    }

    let size = texture.size();
    let len = (size.w as usize)
        .checked_mul(size.h as usize)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .context("snapshot is too large")?;
    let mut data = vec![0u8; len];
    let mut local = texture.clone();
    let mut target = renderer.bind(&mut local)?;
    let mut frame = renderer.render(
        &mut target,
        size.to_logical(1, Transform::Normal).to_physical(1),
        Transform::Normal,
    )?;
    // SAFETY: The current frame owns a bound framebuffer of `size`. The destination
    // allocation uses checked width * height * bytes_per_pixel arithmetic. PACK_ALIGNMENT
    // is 1 and all row-length/skip settings are zero, so ReadPixels writes exactly `len`
    // bytes; no pixel pack buffer is bound, making the pointer refer to our owned Vec.
    let error = frame.with_context(|gl| unsafe {
        let mut pack_buffer = 0;
        let mut read_buffer = 0;
        let mut pack_alignment = 0;
        let mut pack_row_length = 0;
        let mut pack_skip_rows = 0;
        let mut pack_skip_pixels = 0;
        gl.GetIntegerv(ffi::PIXEL_PACK_BUFFER_BINDING, &mut pack_buffer);
        gl.GetIntegerv(ffi::READ_BUFFER, &mut read_buffer);
        gl.GetIntegerv(ffi::PACK_ALIGNMENT, &mut pack_alignment);
        gl.GetIntegerv(ffi::PACK_ROW_LENGTH, &mut pack_row_length);
        gl.GetIntegerv(ffi::PACK_SKIP_ROWS, &mut pack_skip_rows);
        gl.GetIntegerv(ffi::PACK_SKIP_PIXELS, &mut pack_skip_pixels);
        gl.BindBuffer(ffi::PIXEL_PACK_BUFFER, 0);
        gl.ReadBuffer(ffi::COLOR_ATTACHMENT0);
        gl.PixelStorei(ffi::PACK_ALIGNMENT, 1);
        gl.PixelStorei(ffi::PACK_ROW_LENGTH, 0);
        gl.PixelStorei(ffi::PACK_SKIP_ROWS, 0);
        gl.PixelStorei(ffi::PACK_SKIP_PIXELS, 0);
        gl.GetError();
        gl.ReadPixels(
            0,
            0,
            size.w,
            size.h,
            read_format,
            read_type,
            data.as_mut_ptr().cast(),
        );
        let error = gl.GetError();
        gl.BindBuffer(ffi::PIXEL_PACK_BUFFER, pack_buffer as u32);
        gl.ReadBuffer(read_buffer as u32);
        gl.PixelStorei(ffi::PACK_ALIGNMENT, pack_alignment);
        gl.PixelStorei(ffi::PACK_ROW_LENGTH, pack_row_length);
        gl.PixelStorei(ffi::PACK_SKIP_ROWS, pack_skip_rows);
        gl.PixelStorei(ffi::PACK_SKIP_PIXELS, pack_skip_pixels);
        error
    })?;
    // ReadPixels into CPU memory has already completed all preceding rendering.
    let _sync = frame.finish()?;
    anyhow::ensure!(
        error == ffi::NO_ERROR,
        "error reading HDR snapshot pixels: {error:#x}"
    );
    Ok(PortablePixels {
        data: data.into(),
        format,
        size,
        flipped: texture.is_y_inverted(),
        textures: Mutex::new(HashMap::new()),
    })
}

fn download_pixels<R: NiriCaptureRenderer>(
    renderer: &mut R,
    texture: &R::TextureId,
    mut flipped: bool,
) -> anyhow::Result<PortablePixels> {
    let format = texture
        .format()
        .context("snapshot texture has no pixel format")?;
    let size = texture.size();
    let mapping = match renderer.copy_texture(texture, Rectangle::from_size(size), format) {
        Ok(mapping) => mapping,
        Err(_) => {
            flipped = false;
            // External-only textures cannot be read directly. Freeze them into a normal
            // render target first; retain the original format, including HDR precision.
            let buffer = TextureBuffer::from_texture(
                renderer,
                texture.clone(),
                1.,
                Transform::Normal,
                Vec::new(),
            );
            let element = TextureRenderElement::from_texture_buffer(
                buffer,
                (0., 0.),
                1.,
                None,
                None,
                Kind::Unspecified,
            );
            super::render_and_download(
                renderer,
                size.to_logical(1, Transform::Normal).to_physical(1),
                Scale::from(1.),
                Transform::Normal,
                format,
                std::iter::once(element),
            )?
        }
    };
    // copy_texture returns raw rows: preserve the source texture orientation, not the
    // framebuffer mapping orientation (which is always flipped for GLES).
    let format = smithay::backend::renderer::TextureMapping::format(&mapping);
    let data = renderer.map_texture(&mapping)?.into();
    Ok(PortablePixels {
        data,
        format,
        size,
        flipped,
        textures: Mutex::new(HashMap::new()),
    })
}

#[cfg(test)]
mod tests {
    use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
    use smithay::backend::renderer::ExportMem as _;

    use super::*;

    fn renderer() -> GlesRenderer {
        let mut devices = EGLDevice::enumerate().unwrap().collect::<Vec<_>>();
        devices.sort_by_key(|device| !device.is_software());
        devices
            .into_iter()
            .find_map(|device| {
                let display = unsafe { EGLDisplay::new(device) }.ok()?;
                let context = EGLContext::new(&display).ok()?;
                unsafe { GlesRenderer::new(context) }.ok()
            })
            .expect("these rendering tests require an EGL device (Mesa software rendering works)")
    }

    // Different rows and transparent pixels expose orientation and alpha regressions.
    const PIXELS: [u8; 16] = [
        255, 0, 0, 255, 0, 128, 0, 128, 0, 0, 255, 255, 64, 64, 64, 64,
    ];

    fn rendered(renderer: &mut GlesRenderer, buffer: TextureBuffer<TtyOffscreen>) -> Vec<u8> {
        let element = UniversalTextureRenderElement(TextureRenderElement::from_texture_buffer(
            buffer,
            (0., 0.),
            1.,
            None,
            None,
            Kind::Unspecified,
        ));
        let mapping = crate::render_helpers::render_and_download(
            renderer,
            (2, 2).into(),
            Scale::from(1.),
            Transform::Normal,
            Fourcc::Abgr8888,
            std::iter::once(element),
        )
        .unwrap();
        renderer.map_texture(&mapping).unwrap().to_vec()
    }

    #[test]
    #[ignore = "requires an EGL render device (Mesa software rendering works)"]
    fn frozen_texture_migrates_between_contexts_and_caches_the_import() {
        let mut source = renderer();
        let mut target = renderer();
        set_texture_portability(&mut source, true);
        for flipped in [false, true] {
            let texture = source
                .import_memory(&PIXELS, Fourcc::Abgr8888, (2, 2).into(), flipped)
                .unwrap();
            let mut buffer = TextureBuffer::from_texture(
                &source,
                TtyOffscreen::Gles(texture),
                1.,
                Transform::Normal,
                Vec::new(),
            );
            let original = rendered(&mut source, buffer.clone());
            buffer.make_portable(&mut source).unwrap();
            assert_eq!(buffer.portable.as_ref().unwrap().format, Fourcc::Abgr8888);
            assert_eq!(rendered(&mut target, buffer.clone()), original);
            assert_eq!(rendered(&mut target, buffer.clone()), original);
            assert_eq!(
                buffer
                    .portable
                    .as_ref()
                    .unwrap()
                    .textures
                    .lock()
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(rendered(&mut source, buffer.clone()), original);
            // Returning to the source context still uses the original native texture.
            assert_eq!(
                buffer
                    .portable
                    .as_ref()
                    .unwrap()
                    .textures
                    .lock()
                    .unwrap()
                    .len(),
                1
            );
        }
    }

    #[test]
    #[ignore = "requires an EGL render device (Mesa software rendering works)"]
    fn memory_texture_migrates_without_readback_and_disabled_snapshots_stay_local() {
        let mut source = renderer();
        let mut target = renderer();
        set_texture_portability(&mut source, true);
        let buffer = TextureBuffer::from_memory(
            &mut source,
            &PIXELS,
            Fourcc::Abgr8888,
            (2, 2),
            false,
            1.,
            Transform::Normal,
            Vec::new(),
        )
        .unwrap()
        .map_texture(TtyOffscreen::Gles);
        let original = rendered(&mut source, buffer.clone());
        assert_eq!(rendered(&mut target, buffer.clone()), original);

        set_texture_portability(&mut source, false);
        let mut local = TextureBuffer::from_texture(
            &source,
            buffer.texture.clone(),
            1.,
            Transform::Normal,
            Vec::new(),
        );
        local.make_portable(&mut source).unwrap();
        assert!(local.portable.is_none());
        assert_eq!(rendered(&mut source, local.clone()), original);
        assert!(local.texture_for_renderer(&mut target).is_err());
    }

    #[test]
    #[ignore = "requires an EGL render device with RGBA16F (Mesa software rendering works)"]
    fn hdr_frozen_texture_keeps_all_half_float_pixels() {
        let mut source = renderer();
        let mut target = renderer();
        set_texture_portability(&mut source, true);
        // Four RGBA half-float pixels, including values above SDR white.
        let pixels: Vec<u8> = [
            0x4000u16, 0, 0, 0x3c00, 0, 0x4400, 0, 0x3c00, 0, 0, 0x4800, 0x3c00, 0x3800, 0x3800,
            0x3800, 0x3800,
        ]
        .into_iter()
        .flat_map(u16::to_ne_bytes)
        .collect();
        let texture = source
            .import_memory(&pixels, Fourcc::Abgr16161616f, (2, 2).into(), false)
            .unwrap();
        let mut buffer = TextureBuffer::from_texture(
            &source,
            TtyOffscreen::Gles(texture),
            1.,
            Transform::Normal,
            Vec::new(),
        );
        buffer.make_portable(&mut source).unwrap();
        let backing = buffer.portable.as_ref().unwrap();
        assert_eq!(backing.format, Fourcc::Abgr16161616f);
        assert_eq!(backing.data.as_ref(), pixels.as_slice());
        let TtyOffscreen::Gles(texture) = buffer.texture_for_renderer(&mut target).unwrap() else {
            panic!("expected GLES texture")
        };
        let downloaded = download_gles_pixels(&mut target, &texture).unwrap();
        assert_eq!(downloaded.format, Fourcc::Abgr16161616f);
        assert_eq!(downloaded.data.as_ref(), pixels.as_slice());
    }
}
