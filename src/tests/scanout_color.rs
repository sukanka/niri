//! Scanout color state must describe the surfaces eligible for this frame, including subsurfaces.

#![allow(clippy::mutable_key_type)] // Render element IDs have stable equality and hashes.

use std::collections::HashSet;

use smithay::backend::drm::{Curve1DType, ScanoutColorTransform};
use smithay::backend::renderer::element::Id;
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_manager_v1::{
    Primaries, RenderIntent, TransferFunction,
};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface as ServerSurface;
use smithay::reexports::wayland_server::Resource as _;
use smithay::wayland::compositor::get_children;
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::Proxy as _;

use super::client::ClientId;
use super::Fixture;

fn fixture() -> Fixture {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    f
}

/// Returns both ends of a mapped surface. All windows in each test belong to one client,
/// so their protocol IDs identify the matching server resources unambiguously.
fn mapped_window(f: &mut Fixture, client: ClientId) -> (WlSurface, ServerSurface) {
    let window = f.client(client).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(client);
    let window = f.client(client).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(client);

    let server_surface = f
        .niri()
        .layout
        .windows()
        .map(|(_, window)| window.toplevel().wl_surface())
        .find(|server| server.id().protocol_id() == surface.id().protocol_id())
        .expect("window must be mapped")
        .clone();
    (surface, server_surface)
}

fn attach_pq_description(f: &mut Fixture, client: ClientId, surface: &WlSurface) {
    // The existing client helper supplies MaxCLL = 1000 cd/m².
    f.client(client).create_and_attach_hdr_description(
        surface,
        TransferFunction::St2084Pq,
        Primaries::Bt2020,
        RenderIntent::Perceptual,
    );
    f.roundtrip(client);
    surface.commit();
    f.double_roundtrip(client);
}

#[test]
fn only_candidate_windows_contribute_scanout_color_state() {
    let mut f = fixture();
    let client = f.add_client();
    let (_, first) = mapped_window(&mut f, client);
    let (_, second) = mapped_window(&mut f, client);
    let output = f.niri_output(1);
    let first = Id::from_wayland_resource(&first);
    let second = Id::from_wayland_resource(&second);

    let candidates = HashSet::from([first.clone()]);
    let transforms = f
        .niri()
        .scanout_color_transforms(&output, &candidates, false, 203., 203.);
    assert_eq!(transforms.len(), 1);
    assert_eq!(
        transforms.get(&first),
        Some(&Some(ScanoutColorTransform::IDENTITY))
    );
    assert!(!transforms.contains_key(&second));

    // Moving between workspaces can replace all candidates without changing their colors.
    let candidates = HashSet::from([second.clone()]);
    let transforms = f
        .niri()
        .scanout_color_transforms(&output, &candidates, false, 203., 203.);
    assert_eq!(transforms.len(), 1);
    assert!(!transforms.contains_key(&first));
    assert_eq!(
        transforms.get(&second),
        Some(&Some(ScanoutColorTransform::IDENTITY))
    );

    assert!(f
        .niri()
        .scanout_color_transforms(&output, &HashSet::new(), true, 203., 800.)
        .is_empty());
}

#[test]
fn non_candidate_color_changes_do_not_change_scanout_state() {
    let mut f = fixture();
    let client = f.add_client();
    let (_, candidate) = mapped_window(&mut f, client);
    let (other_client_surface, other) = mapped_window(&mut f, client);
    let output = f.niri_output(1);
    let candidate = Id::from_wayland_resource(&candidate);
    let other = Id::from_wayland_resource(&other);
    let candidates = HashSet::from([candidate.clone()]);

    let before = f
        .niri()
        .scanout_color_transforms(&output, &candidates, true, 203., 800.);
    let sdr_to_hdr = before[&candidate].expect("SDR must retain its HDR scanout conversion");
    assert_eq!(sdr_to_hdr.decode, Some(Curve1DType::Gamma22));
    assert_eq!(sdr_to_hdr.encode, Some(Curve1DType::Pq125InvEotf));

    attach_pq_description(&mut f, client, &other_client_surface);

    let after = f
        .niri()
        .scanout_color_transforms(&output, &candidates, true, 203., 800.);
    assert_eq!(before, after);
    assert!(!after.contains_key(&other));

    // Confirm that the protocol update really committed: including the other window now
    // must deny its scanout because its 1000-nit content requires tone mapping.
    let both = HashSet::from([candidate, other.clone()]);
    let transforms = f
        .niri()
        .scanout_color_transforms(&output, &both, true, 203., 800.);
    assert_eq!(transforms.len(), 2);
    assert_eq!(transforms.get(&other), Some(&None));
}

#[test]
fn candidate_subsurface_is_kept_when_its_parent_is_not_a_candidate() {
    let mut f = fixture();
    let client = f.add_client();
    let (parent, server_parent) = mapped_window(&mut f, client);
    let child = f.client(client).create_committed_subsurface(&parent);
    f.roundtrip(client);
    attach_pq_description(&mut f, client, &child);
    // Subsurface color state is synchronized with the parent's commit.
    parent.commit();
    f.double_roundtrip(client);

    let server_child = get_children(&server_parent)
        .into_iter()
        .find(|server| server.id().protocol_id() == child.id().protocol_id())
        .expect("subsurface must belong to the mapped parent");
    let parent_id = Id::from_wayland_resource(&server_parent);
    let child_id = Id::from_wayland_resource(&server_child);
    let candidates = HashSet::from([child_id.clone()]);
    let output = f.niri_output(1);

    let transforms = f
        .niri()
        .scanout_color_transforms(&output, &candidates, true, 203., 2000.);
    assert_eq!(transforms.len(), 1);
    assert!(!transforms.contains_key(&parent_id));
    assert_eq!(
        transforms.get(&child_id),
        Some(&Some(ScanoutColorTransform::IDENTITY)),
        "Proton-style PQ subsurfaces must retain HDR passthrough"
    );
}

#[test]
fn tone_mapping_denial_is_kept_for_candidate_on_hdr_and_sdr_outputs() {
    let mut f = fixture();
    let client = f.add_client();
    let (surface, server_surface) = mapped_window(&mut f, client);
    attach_pq_description(&mut f, client, &surface);
    let surface_id = Id::from_wayland_resource(&server_surface);
    let candidates = HashSet::from([surface_id.clone()]);
    let output = f.niri_output(1);

    for (blend_hdr, peak) in [(true, 800.), (false, 203.)] {
        let transforms =
            f.niri()
                .scanout_color_transforms(&output, &candidates, blend_hdr, 203., peak);
        assert_eq!(transforms.len(), 1);
        assert_eq!(
            transforms.get(&surface_id),
            Some(&None),
            "a tone-mapped surface needs an explicit denial, especially with SDR deny_unlisted=false"
        );
    }

    // A changed output peak can make the same committed surface eligible again.
    let transforms = f
        .niri()
        .scanout_color_transforms(&output, &candidates, true, 203., 2000.);
    assert_eq!(
        transforms.get(&surface_id),
        Some(&Some(ScanoutColorTransform::IDENTITY))
    );
}

#[test]
fn scrgb_candidate_keeps_its_conversion_on_an_sdr_output() {
    let mut f = fixture();
    let client = f.add_client();
    let (surface, server_surface) = mapped_window(&mut f, client);
    f.client(client)
        .create_and_attach_scrgb_description(&surface);
    f.roundtrip(client);
    surface.commit();
    f.double_roundtrip(client);

    let surface_id = Id::from_wayland_resource(&server_surface);
    let candidates = HashSet::from([surface_id.clone()]);
    let output = f.niri_output(1);
    let transforms = f
        .niri()
        .scanout_color_transforms(&output, &candidates, false, 203., 203.);
    assert_eq!(transforms.len(), 1);
    let transform = transforms[&surface_id].expect("scRGB requires a conversion on SDR");
    assert_eq!(transform.decode, None);
    assert_eq!(transform.ctm, None);
    assert_eq!(transform.encode, Some(Curve1DType::Gamma22Inv));
    assert!((transform.multiplier - 80. / 203.).abs() < 1e-9);
    assert!(!transform.is_identity());
}
