use smithay::reexports::wayland_protocols::wp::color_representation::v1::client::{
    wp_color_representation_manager_v1::Event,
    wp_color_representation_surface_v1::{AlphaMode, Coefficients, Range, WpColorRepresentationSurfaceV1},
};
use wayland_client::WEnum;

use super::{client, Fixture};

wayland_client::delegate_noop!(client::State: WpColorRepresentationSurfaceV1);

#[test]
fn full_range_rgb_subsurface() {
    let mut fixture = Fixture::new();
    fixture.add_output(1, (800, 600));
    let id = fixture.add_client();
    fixture.double_roundtrip(id);

    let client = fixture.client(id);
    assert!(matches!(
        client.state.color_representation_events.as_slice(),
        [
            Event::SupportedAlphaMode {
                alpha_mode: WEnum::Value(AlphaMode::PremultipliedElectrical)
            },
            Event::SupportedCoefficientsAndRanges {
                coefficients: WEnum::Value(Coefficients::Identity),
                range: WEnum::Value(Range::Full),
            },
            Event::Done,
        ]
    ));
    let manager = client.state.color_representation_manager.clone().unwrap();
    let queue = client.qh.clone();
    let parent = client.create_window().surface.clone();
    let surface = client.create_committed_subsurface(&parent);

    let representation = manager.get_surface(&surface, &queue, ());
    representation.set_alpha_mode(AlphaMode::PremultipliedElectrical);
    representation.set_coefficients_and_range(Coefficients::Identity, Range::Full);
    surface.commit();
    parent.commit();
    fixture.double_roundtrip(id);

    representation.destroy();
    let replacement = manager.get_surface(&surface, &queue, ());
    replacement.set_coefficients_and_range(Coefficients::Identity, Range::Full);
    surface.commit();
    parent.commit();
    fixture.double_roundtrip(id);
}
