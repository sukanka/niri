use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::utils::{import_surface, RendererSurfaceStateUserData};
use smithay::backend::renderer::{ImportAll, Renderer, Texture as _};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Physical, Point, Scale, Transform};
use smithay::wayland::color::management::ColorManagementSurfaceCachedState;
use smithay::wayland::compositor::{with_surface_tree_downward, TraversalAction};

use super::blend::{BlendSurfaceRenderElement, ContentColor};
use super::renderer::NiriRenderer;
use super::texture::{TextureBuffer, TextureRenderElement};
use super::BakedBuffer;
use crate::backend::tty_renderer::TtyOffscreen;

/// Renders elements from a surface tree as textures into `storage`.
pub fn render_snapshot_from_surface_tree<R: NiriRenderer>(
    renderer: &mut R,
    surface: &WlSurface,
    location: Point<f64, Logical>,
    storage: &mut Vec<BakedBuffer<TextureBuffer<TtyOffscreen>>>,
) {
    let _span = tracy_client::span!("render_snapshot_from_surface_tree");

    with_surface_tree_downward(
        surface,
        location,
        |_, states, location| {
            let mut location = *location;
            let data = states.data_map.get::<RendererSurfaceStateUserData>();

            if let Some(data) = data {
                let data = &*data.lock().unwrap();

                if let Some(view) = data.view() {
                    location += view.offset.to_f64();
                    TraversalAction::DoChildren(location)
                } else {
                    TraversalAction::SkipChildren
                }
            } else {
                TraversalAction::SkipChildren
            }
        },
        |_, states, location| {
            let mut location = *location;
            let data = states.data_map.get::<RendererSurfaceStateUserData>();

            if let Some(data) = data {
                let Some(view) = data.lock().unwrap().view() else {
                    return;
                };
                location += view.offset.to_f64();

                if let Err(err) = import_surface(renderer, states) {
                    warn!("failed to import surface: {err:?}");
                    return;
                }

                let data = data.lock().unwrap();
                let Some(texture) = data.texture(renderer.context_id()) else {
                    return;
                };

                let Some(source_buffer) = data.buffer() else {
                    return;
                };
                // Freeze every snapshot, including on a single GPU. A cloned imported texture
                // still aliases client memory and cannot outlive this commit's Buffer or acquire
                // a different submission's release fence when an animation samples it later.
                let texture = match freeze_surface_texture(renderer, texture, source_buffer) {
                    Ok(texture) => texture,
                    Err(err) => {
                        warn!("error freezing surface snapshot: {err:#}");
                        return;
                    }
                };
                let mut buffer = TextureBuffer::from_texture(
                    renderer,
                    texture,
                    f64::from(data.buffer_scale()),
                    data.buffer_transform(),
                    Vec::new(),
                );
                if let Err(err) = buffer.make_portable(renderer) {
                    warn!("error preserving surface snapshot for another GPU: {err:#}");
                }

                let baked = BakedBuffer {
                    buffer,
                    location,
                    src: Some(view.src),
                    dst: Some(view.dst),
                };

                storage.push(baked);
            }
        },
        |_, _, _| true,
    );
}

fn freeze_surface_texture<R: NiriRenderer>(
    renderer: &mut R,
    texture: &R::TextureId,
    source_buffer: &smithay::backend::renderer::utils::Buffer,
) -> anyhow::Result<TtyOffscreen> {
    // Go through the full renderer, including MultiFrame's synchronization and import
    // preparation. Native texture-cache access can miss DMA fences or deferred CPU uploads.
    let source =
        TextureBuffer::from_texture(renderer, texture.clone(), 1., Transform::Normal, Vec::new());
    let element = TextureRenderElement::from_texture_buffer(
        source,
        (0., 0.),
        1.,
        None,
        None,
        Kind::Unspecified,
    );
    let (texture, _sync) = source_buffer.with_read_source(|| {
        super::render_to_texture(
            renderer,
            texture
                .size()
                .to_logical(1, Transform::Normal)
                .to_physical(1),
            Scale::from(1.),
            Transform::Normal,
            texture
                .format()
                .unwrap_or(smithay::backend::allocator::Fourcc::Abgr8888),
            std::iter::once(element),
        )
    })?;
    Ok(texture)
}

pub fn push_elements_from_surface_tree<R>(
    renderer: &mut R,
    surface: &WlSurface,
    // Fractional scale expects surface buffers to be aligned to physical pixels.
    location: Point<i32, Physical>,
    scale: Scale<f64>,
    alpha: f32,
    kind: Kind,
    push: &mut dyn FnMut(BlendSurfaceRenderElement<R>),
) where
    R: Renderer + ImportAll,
    R::TextureId: Clone + 'static,
{
    let _span = tracy_client::span!("push_elements_from_surface_tree");

    let location = location.to_f64();

    with_surface_tree_downward(
        surface,
        location,
        |_, states, location| {
            let mut location = *location;
            let data = states.data_map.get::<RendererSurfaceStateUserData>();

            if let Some(data) = data {
                if let Some(view) = data.lock().unwrap().view() {
                    location += view.offset.to_f64().to_physical(scale);
                    TraversalAction::DoChildren(location)
                } else {
                    TraversalAction::SkipChildren
                }
            } else {
                TraversalAction::SkipChildren
            }
        },
        |surface, states, location| {
            let mut location = *location;
            let data = states.data_map.get::<RendererSurfaceStateUserData>();

            if let Some(data) = data {
                let has_view = if let Some(view) = data.lock().unwrap().view() {
                    location += view.offset.to_f64().to_physical(scale);
                    true
                } else {
                    false
                };

                if has_view {
                    // Content carrying an HDR image description is already encoded in the
                    // output blend space and must not be re-encoded when composited;
                    // Windows-scRGB content instead needs the dedicated absolute encode.
                    let content = ContentColor::from_description(
                        states
                            .cached_state
                            .get::<ColorManagementSurfaceCachedState>()
                            .current()
                            .description,
                    );

                    match WaylandSurfaceRenderElement::from_surface(
                        renderer, surface, states, location, alpha, kind,
                    ) {
                        Ok(Some(surface)) => push(BlendSurfaceRenderElement::new(surface, content)),
                        Ok(None) => {} // surface is not mapped
                        Err(err) => {
                            warn!("failed to import surface: {}", err);
                        }
                    };
                }
            }
        },
        |_, _, _| true,
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smithay::backend::allocator::Fourcc;
    use smithay::backend::egl::native::EGLSurfacelessDisplay;
    use smithay::backend::egl::{EGLContext, EGLDisplay};
    use smithay::backend::renderer::gles::GlesRenderer;
    use smithay::backend::renderer::{ExportMem as _, ImportMem as _};
    use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
    use smithay::reexports::wayland_server::{self, Resource};
    use smithay::utils::Rectangle;

    use super::*;

    struct State;
    impl wayland_server::Dispatch<WlBuffer, ()> for State {
        fn request(
            _: &mut Self,
            _: &wayland_server::Client,
            _: &WlBuffer,
            _: <WlBuffer as Resource>::Request,
            _: &(),
            _: &wayland_server::DisplayHandle,
            _: &mut wayland_server::DataInit<'_, Self>,
        ) {
        }
    }

    #[test]
    fn surface_snapshot_freezes_client_pixels_before_delayed_animation_reads() {
        let renderer = (|| -> anyhow::Result<_> {
            let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay)? };
            let context = EGLContext::new(&display)?;
            Ok(unsafe { GlesRenderer::new(context)? })
        })();
        let mut renderer = match renderer {
            Ok(renderer) => renderer,
            Err(err) => {
                if std::env::var_os("NIRI_TEST_REQUIRE_GPU")
                    .is_some_and(|v| !v.is_empty() && v != "0")
                {
                    panic!("snapshot test requires GLES: {err:#}");
                }
                eprintln!("skipping snapshot test: {err:#}");
                return;
            }
        };
        let display = wayland_server::Display::<State>::new().unwrap();
        let (_socket, server) = std::os::unix::net::UnixStream::pair().unwrap();
        let client = display
            .handle()
            .insert_client(server, Arc::new(()))
            .unwrap();
        let buffer = client
            .create_resource::<WlBuffer, (), State>(&display.handle(), 1, ())
            .unwrap();
        let buffer = smithay::backend::renderer::utils::Buffer::with_implicit(buffer);
        let source = renderer
            .import_memory(&[255, 0, 0, 255], Fourcc::Abgr8888, (1, 1).into(), false)
            .unwrap();
        let snapshot = freeze_surface_texture(&mut renderer, &source, &buffer)
            .unwrap()
            .into_gles()
            .unwrap();
        assert_ne!(snapshot.tex_id(), source.tex_id());
        renderer
            .update_memory(
                &source,
                &[0, 255, 0, 255],
                Rectangle::from_size((1, 1).into()),
            )
            .unwrap();
        drop(buffer);
        drop(source);
        let mapping = renderer
            .copy_texture(
                &snapshot,
                Rectangle::from_size((1, 1).into()),
                Fourcc::Abgr8888,
            )
            .unwrap();
        assert_eq!(renderer.map_texture(&mapping).unwrap(), [255, 0, 0, 255]);
    }
}
