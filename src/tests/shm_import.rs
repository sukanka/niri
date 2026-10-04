use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::utils::RendererSurfaceStateUserData;
use smithay::backend::renderer::{Color32F, Frame as _};
use smithay::utils::{Rectangle, Transform};
use smithay::wayland::compositor::with_states;
use wayland_client::protocol::wl_shm::Format;

use super::{gpu, Fixture};
use crate::render_helpers::renderer::NiriCaptureRenderer;
use crate::render_helpers::{copy_framebuffer, create_texture};

fn check_format_changes<R: NiriCaptureRenderer>(renderer: &mut R) {
    let mut fixture = Fixture::new();
    fixture.add_output(1, (100, 100));
    let client = fixture.add_client();
    let window = fixture.client(client).create_window();
    let surface = window.surface.clone();
    window.commit();
    fixture.roundtrip(client);
    let window = fixture.client(client).window(&surface);
    window.attach_shm_pixels(1, 1, Format::Xrgb8888, &[0, 0, 255, 0]);
    window.ack_last_and_commit();
    fixture.double_roundtrip(client);
    let server_surface = fixture
        .niri()
        .layout
        .focus()
        .unwrap()
        .toplevel()
        .wl_surface()
        .clone();

    for (format, source, expected) in [
        (Format::Xrgb8888, [0, 0, 255, 0], [255, 0, 0, 255]),
        (Format::Argb8888, [0, 0, 128, 128], [128, 0, 0, 128]),
        (Format::Abgr8888, [128, 0, 0, 128], [128, 0, 0, 128]),
        (Format::Xbgr8888, [255, 0, 0, 0], [255, 0, 0, 255]),
    ] {
        let window = fixture.client(client).window(&surface);
        window.attach_shm_pixels(1, 1, format, &source);
        window.commit();
        fixture.double_roundtrip(client);
        let rect = Rectangle::from_size((1, 1).into());
        let texture = with_states(&server_surface, |states| {
            let data = states
                .data_map
                .get::<RendererSurfaceStateUserData>()
                .unwrap()
                .lock()
                .unwrap();
            renderer
                .import_buffer(data.buffer().unwrap(), Some(states), &[rect])
                .unwrap()
                .unwrap()
        });
        let rect = Rectangle::from_size((1, 1).into());
        let mut destination = create_texture(renderer, (1, 1).into(), Fourcc::Abgr8888).unwrap();
        let mut target = renderer.bind(&mut destination).unwrap();
        let mut frame = renderer
            .render(&mut target, (1, 1).into(), Transform::Normal)
            .unwrap();
        frame.clear(Color32F::TRANSPARENT, &[rect]).unwrap();
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size((1., 1.).into()),
                rect,
                &[rect],
                &[],
                Transform::Normal,
                1.,
            )
            .unwrap();
        let _sync = frame.finish().unwrap();
        let mapping = copy_framebuffer(renderer, &target, Fourcc::Abgr8888).unwrap();
        assert_eq!(
            renderer.map_texture(&mapping).unwrap(),
            expected,
            "format {format:?}"
        );
    }
}

#[test]
fn shm_format_changes_gles() {
    if let Some(mut renderer) = gpu::gles_renderer() {
        check_format_changes(&mut renderer);
    }
}

#[test]
fn shm_format_changes_vulkan() {
    if let Some(mut renderer) = gpu::vulkan_renderer() {
        check_format_changes(&mut renderer);
    }
}
