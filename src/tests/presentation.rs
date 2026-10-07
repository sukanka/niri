use std::sync::{Arc, Mutex};
use std::time::Duration;

use smithay::desktop::utils::SurfacePresentationFeedback;
use smithay::reexports::wayland_protocols::wp::presentation_time::client::wp_presentation_feedback::{
    self, WpPresentationFeedback,
};
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::Kind;
use smithay::utils::{ClockSource as _, Monotonic};
use smithay::wayland::compositor::with_states;
use smithay::wayland::presentation::Refresh;
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, Proxy as _, QueueHandle};

use super::client::{ClientId, State as ClientState};
use super::Fixture;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeedbackEvent {
    Presented,
    Discarded,
}

type Feedback = Arc<Mutex<Option<FeedbackEvent>>>;

impl Dispatch<WpPresentationFeedback, Feedback> for ClientState {
    fn event(
        _state: &mut Self,
        _proxy: &WpPresentationFeedback,
        event: wp_presentation_feedback::Event,
        data: &Feedback,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            wp_presentation_feedback::Event::Presented { .. } => {
                *data.lock().unwrap() = Some(FeedbackEvent::Presented);
            }
            wp_presentation_feedback::Event::Discarded => {
                *data.lock().unwrap() = Some(FeedbackEvent::Discarded);
            }
            _ => (),
        }
    }
}

fn set_up() -> (Fixture, ClientId, WlSurface) {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(100, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);
    (f, id, surface)
}

fn request_feedback(f: &mut Fixture, id: ClientId, surface: &WlSurface) -> Feedback {
    let client = f.client(id);
    let data = Feedback::default();
    client
        .state
        .presentation
        .as_ref()
        .unwrap()
        .feedback(surface, &client.qh, data.clone());
    data
}

#[test]
fn unmap_discards_feedback_without_destroying_surface() {
    let (mut f, id, surface) = set_up();
    let feedback = request_feedback(&mut f, id, &surface);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.commit();
    window.attach_null();
    window.commit();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(surface.is_alive());
}

#[test]
fn unmap_discards_subsurface_feedback() {
    let (mut f, id, surface) = set_up();
    let child = f.client(id).create_committed_subsurface(&surface);
    f.double_roundtrip(id);

    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    let window = f.client(id).window(&surface);
    window.commit();
    window.attach_null();
    window.commit();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(child.is_alive());
}

#[test]
fn destroying_toplevel_discards_feedback_without_destroying_surface() {
    let (mut f, id, surface) = set_up();
    let feedback = request_feedback(&mut f, id, &surface);
    let window = f.client(id).window(&surface);
    window.commit();
    window.xdg_toplevel.destroy();
    window.xdg_surface.destroy();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(surface.is_alive());
}

#[test]
fn destroying_toplevel_discards_subsurface_feedback() {
    let (mut f, id, surface) = set_up();
    let child = f.client(id).create_committed_subsurface(&surface);
    f.double_roundtrip(id);

    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    let window = f.client(id).window(&surface);
    window.commit();
    window.xdg_toplevel.destroy();
    window.xdg_surface.destroy();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(child.is_alive());
}

#[test]
fn unmap_preserves_feedback_owned_by_a_submitted_frame() {
    let (mut f, id, surface) = set_up();
    let submitted = request_feedback(&mut f, id, &surface);
    f.client(id).window(&surface).commit();
    f.double_roundtrip(id);

    let server_surface = f
        .niri()
        .layout
        .windows()
        .next()
        .unwrap()
        .1
        .window
        .toplevel()
        .unwrap()
        .wl_surface()
        .clone();
    let mut frame_feedback = with_states(&server_surface, |states| {
        SurfacePresentationFeedback::from_states(states, Kind::empty()).unwrap()
    });

    let unsubmitted = request_feedback(&mut f, id, &surface);
    let window = f.client(id).window(&surface);
    window.commit();
    window.attach_null();
    window.commit();
    f.double_roundtrip(id);

    assert_eq!(*unsubmitted.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert_eq!(*submitted.lock().unwrap(), None);

    frame_feedback.presented(
        &f.niri_output(1),
        Monotonic::ID as u32,
        Duration::ZERO,
        Refresh::Unknown,
        1,
        Kind::empty(),
    );
    f.double_roundtrip(id);
    assert_eq!(*submitted.lock().unwrap(), Some(FeedbackEvent::Presented));
}

#[test]
fn feedback_requested_after_unmap_survives_remapping() {
    let (mut f, id, surface) = set_up();
    let window = f.client(id).window(&surface);
    window.attach_null();
    window.commit();
    let feedback = request_feedback(&mut f, id, &surface);
    f.double_roundtrip(id);
    assert_eq!(*feedback.lock().unwrap(), None);

    f.client(id).window(&surface).commit();
    f.double_roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);
    assert_ne!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
}

#[test]
fn unmap_discards_feedback_from_detached_subsurfaces() {
    let (mut f, id, surface) = set_up();
    let (child, role) = f.client(id).create_committed_subsurface_with_role(&surface);
    role.set_desync();
    f.double_roundtrip(id);

    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    role.destroy();
    // Vulkan WSI may commit once more while retiring the detached surface.
    child.commit();
    let window = f.client(id).window(&surface);
    window.attach_null();
    window.commit();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(child.is_alive());
}

#[test]
fn destroying_toplevel_discards_feedback_from_detached_subsurfaces() {
    let (mut f, id, surface) = set_up();
    let (child, role) = f.client(id).create_committed_subsurface_with_role(&surface);
    role.set_desync();
    f.double_roundtrip(id);

    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    role.destroy();
    let window = f.client(id).window(&surface);
    window.xdg_toplevel.destroy();
    window.xdg_surface.destroy();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(child.is_alive());
}

#[test]
fn destroying_parent_surface_discards_detached_subsurface_feedback() {
    let (mut f, id, surface) = set_up();
    let (child, role) = f.client(id).create_committed_subsurface_with_role(&surface);
    role.set_desync();
    f.double_roundtrip(id);
    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    role.destroy();
    surface.destroy();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(child.is_alive());
}

#[test]
fn reparented_subsurface_feedback_survives_old_window_unmap() {
    let (mut f, id, surface) = set_up();
    let window = f.client(id).create_window();
    let new_parent = window.surface.clone();
    window.commit();
    f.double_roundtrip(id);
    let window = f.client(id).window(&new_parent);
    window.attach_new_buffer();
    window.set_size(100, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let (child, old_role) = f.client(id).create_committed_subsurface_with_role(&surface);
    old_role.set_desync();
    f.double_roundtrip(id);
    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    old_role.destroy();
    let client = f.client(id);
    let _new_role = client.state.subcompositor.as_ref().unwrap().get_subsurface(
        &child,
        &new_parent,
        &client.qh,
        (),
    );
    let window = f.client(id).window(&surface);
    window.attach_null();
    window.commit();
    f.double_roundtrip(id);

    assert_ne!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
}

#[test]
fn window_destruction_retires_fifo_blocked_detached_subsurface_feedback() {
    let (mut f, id, surface) = set_up();
    let (child, role) = f.client(id).create_committed_subsurface_with_role(&surface);
    role.set_desync();
    f.double_roundtrip(id);
    let client = f.client(id);
    let fifo = client
        .state
        .fifo_manager
        .as_ref()
        .unwrap()
        .get_fifo(&child, &client.qh, ());

    fifo.set_barrier();
    child.commit();
    fifo.wait_barrier();
    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    role.destroy();
    let window = f.client(id).window(&surface);
    window.xdg_toplevel.destroy();
    window.xdg_surface.destroy();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(child.is_alive());
}

#[test]
fn window_destruction_retires_expired_timer_on_detached_subsurface() {
    let (mut f, id, surface) = set_up();
    let (child, role) = f.client(id).create_committed_subsurface_with_role(&surface);
    role.set_desync();
    f.double_roundtrip(id);
    let client = f.client(id);
    let timer = client
        .state
        .commit_timing_manager
        .as_ref()
        .unwrap()
        .get_timer(&child, &client.qh, ());
    let timestamp = crate::utils::get_monotonic_time().saturating_sub(Duration::from_secs(1));
    timer.set_timestamp(
        (timestamp.as_secs() >> 32) as u32,
        timestamp.as_secs() as u32,
        timestamp.subsec_nanos(),
    );
    let feedback = request_feedback(&mut f, id, &child);
    child.commit();
    role.destroy();
    let window = f.client(id).window(&surface);
    window.xdg_toplevel.destroy();
    window.xdg_surface.destroy();
    f.double_roundtrip(id);

    assert_eq!(*feedback.lock().unwrap(), Some(FeedbackEvent::Discarded));
    assert!(child.is_alive());
}
