use super::Fixture;

#[test]
fn headless_render_status_reports_unknown_drm_state() {
    let mut f = Fixture::new();
    f.add_output(2, (1920, 1080));
    f.add_output(1, (1280, 720));

    let state = f.niri_state();
    let outputs = state.backend.render_status(&state.niri);
    assert_eq!(outputs.len(), 2);
    assert!(outputs[0].name < outputs[1].name);
    for output in outputs {
        assert!(output.render_node.is_none());
        assert!(output.scanout_node.is_none());
        assert!(output.cross_gpu_composition.is_none());
        assert!(output.vrr_enabled.is_none());
        assert!(output.hdr_enabled.is_none());
        assert!(output.last_frame.is_none());
    }
}
