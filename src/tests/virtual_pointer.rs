use wayland_client::protocol::wl_pointer;

use super::*;

#[test]
fn axis_discrete_overflow() {
    let mut f = Fixture::new();
    let id = f.add_client();

    let client = f.client(id);
    let manager = client.state.virtual_pointer_manager.as_ref().unwrap();
    let pointer = manager.create_virtual_pointer(None, &client.qh, ());
    pointer.axis_discrete(0, wl_pointer::Axis::VerticalScroll, 0., i32::MAX);
    f.roundtrip(id);
}

fn setup_constraint_test() -> (
    Fixture,
    client::ClientId,
    wayland_client::protocol::wl_surface::WlSurface,
    wl_pointer::WlPointer,
    smithay::reexports::wayland_protocols_wlr::virtual_pointer::v1::client::zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
    smithay::utils::Point<f64, smithay::utils::Logical>,
){
    let mut f = Fixture::new();
    f.add_output(1, (800, 600));
    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(100, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);
    f.niri_complete_animations();
    f.double_roundtrip(id);

    let origin = (0..600)
        .step_by(10)
        .find_map(|y| {
            (0..800).step_by(10).find_map(|x| {
                f.niri()
                    .contents_under((f64::from(x), f64::from(y)).into())
                    .surface
                    .map(|(_, origin)| origin)
            })
        })
        .expect("mapped surface must have an input region");
    let client = f.client(id);
    let pointer = client
        .state
        .seat
        .as_ref()
        .unwrap()
        .get_pointer(&client.qh, ());
    let virtual_pointer = client
        .state
        .virtual_pointer_manager
        .as_ref()
        .unwrap()
        .create_virtual_pointer(None, &client.qh, ());
    f.double_roundtrip(id);
    f.niri_state()
        .move_cursor(origin + smithay::utils::Point::from((25., 25.)));
    f.double_roundtrip(id);
    (f, id, surface, pointer, virtual_pointer, origin)
}

#[test]
fn absolute_motion_respects_pointer_lock() {
    use smithay::reexports::wayland_protocols::wp::pointer_constraints::zv1::client::zwp_pointer_constraints_v1::Lifetime;
    use smithay::wayland::pointer_constraints::with_pointer_constraint;

    let (mut f, id, surface, pointer, virtual_pointer, origin) = setup_constraint_test();
    let client = f.client(id);
    let locked = client
        .state
        .pointer_constraints
        .as_ref()
        .unwrap()
        .lock_pointer(
            &surface,
            &pointer,
            None,
            Lifetime::Persistent,
            &client.qh,
            (),
        );
    f.double_roundtrip(id);
    let server_pointer = f.niri().seat.get_pointer().unwrap();
    let focused = server_pointer.current_focus().unwrap();
    assert!(with_pointer_constraint(
        &focused,
        &server_pointer,
        |constraint| constraint.unwrap().is_active()
    ));
    let initial = server_pointer.current_location();
    virtual_pointer.motion_absolute(1, 700, 500, 800, 600);
    virtual_pointer.frame();
    f.double_roundtrip(id);
    assert_eq!(server_pointer.current_location(), initial);
    assert_eq!(server_pointer.current_focus(), Some(focused));

    locked.destroy();
    f.double_roundtrip(id);
    virtual_pointer.motion_absolute(
        2,
        (origin.x + 60.) as u32,
        (origin.y + 60.) as u32,
        800,
        600,
    );
    virtual_pointer.frame();
    f.double_roundtrip(id);
    assert_eq!(
        server_pointer.current_location(),
        origin + smithay::utils::Point::from((60., 60.))
    );
}

#[test]
fn absolute_and_relative_motion_share_confinement_and_slide() {
    use smithay::reexports::wayland_protocols::wp::pointer_constraints::zv1::client::zwp_pointer_constraints_v1::Lifetime;
    use smithay::wayland::pointer_constraints::with_pointer_constraint;

    let (mut f, id, surface, pointer, virtual_pointer, origin) = setup_constraint_test();
    let client = f.client(id);
    let region = client
        .state
        .compositor
        .as_ref()
        .unwrap()
        .create_region(&client.qh, ());
    region.add(10, 10, 80, 80);
    region.subtract(50, 10, 10, 80);
    let _confined = client
        .state
        .pointer_constraints
        .as_ref()
        .unwrap()
        .confine_pointer(
            &surface,
            &pointer,
            Some(&region),
            Lifetime::Persistent,
            &client.qh,
            (),
        );
    f.double_roundtrip(id);
    let server_pointer = f.niri().seat.get_pointer().unwrap();
    let focused = server_pointer.current_focus().unwrap();
    assert!(with_pointer_constraint(
        &focused,
        &server_pointer,
        |constraint| constraint.unwrap().is_active()
    ));
    virtual_pointer.motion_absolute(
        1,
        (origin.x + 80.) as u32,
        (origin.y + 70.) as u32,
        800,
        600,
    );
    virtual_pointer.frame();
    f.double_roundtrip(id);
    let position = server_pointer.current_location() - origin;
    assert!(position.x < 50. && position.x > 49.99, "{position:?}");
    assert_eq!(position.y, 70.);
    assert_eq!(server_pointer.current_focus(), Some(focused.clone()));

    virtual_pointer.motion(2, 200., -40.);
    virtual_pointer.frame();
    f.double_roundtrip(id);
    let position = server_pointer.current_location() - origin;
    assert!(position.x < 50. && position.x > 49.99, "{position:?}");
    assert_eq!(position.y, 30.);
    assert_eq!(server_pointer.current_focus(), Some(focused));
}
