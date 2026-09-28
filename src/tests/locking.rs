//! Presentation policy must stop following desktop windows as soon as locking begins.

use niri_config::Config;
use smithay::backend::renderer::element::{
    Id, RenderElementPresentationState, RenderElementState, RenderElementStates,
};
use smithay::reexports::wayland_protocols::ext::session_lock::v1::server::ext_session_lock_v1::ExtSessionLockV1;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::Resource as _;

use super::Fixture;
use crate::niri::LockState;

#[test]
fn lock_ignores_desktop_presentation_requests_before_visibility_updates() {
    let config = Config::parse_mem(
        r#"
        output "headless-1" {
            allow-tearing true
            variable-refresh-rate on-demand=true
        }
        window-rule {
            allow-tearing true
            variable-refresh-rate true
        }
        "#,
    )
    .unwrap();
    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let output = f.niri_output(1);
    let niri = f.niri();
    let (_, window) = niri.layout.windows().next().expect("window must be mapped");
    let surface_id = Id::from_wayland_resource(window.toplevel().wl_surface());
    let mut states = RenderElementStates::default();
    states.states.insert(
        surface_id,
        RenderElementState {
            visible_area: 1920 * 1080,
            presentation_state: RenderElementPresentationState::Async,
            needs_capture: false,
        },
    );
    niri.update_primary_scanout_output(&output, &states);
    assert!(niri.output_allows_tearing(&output));
    assert!(niri.output_wants_on_demand_vrr(&output));

    // An inert resource is sufficient: these policies depend on session state, not on
    // locker requests. Keep the desktop's last presented state intact to cover the
    // first lock-screen frame, before rendering has cleared its visibility.
    let lock = ExtSessionLockV1::from_id(&niri.display_handle, ObjectId::null()).unwrap();
    niri.lock_state = LockState::Locked(lock);
    assert!(!niri.output_allows_tearing(&output));
    assert!(!niri.output_wants_on_demand_vrr(&output));

    // Unlocking restores the desktop's requested policy without changing its buffers.
    niri.lock_state = LockState::Unlocked;
    assert!(niri.output_allows_tearing(&output));
    assert!(niri.output_wants_on_demand_vrr(&output));

    // A genuinely hidden desktop window must not keep requesting either policy.
    niri.update_primary_scanout_output(&output, &RenderElementStates::default());
    assert!(!niri.output_allows_tearing(&output));
    assert!(!niri.output_wants_on_demand_vrr(&output));
}
