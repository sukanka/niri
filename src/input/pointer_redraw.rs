//! Output damage caused by ordinary pointer motion.

use smallvec::SmallVec;
use smithay::desktop::utils::bbox_from_surface_tree;
use smithay::desktop::LayerSurface;
use smithay::input::pointer::CursorImageStatus;
use smithay::output::Output;
use smithay::utils::{Logical, Point, Rectangle, Size};

use crate::cursor::RenderCursor;
use crate::niri::Niri;
use crate::window::mapped::MappedId;

pub(super) fn displayed_position(niri: &Niri) -> Option<Point<f64, Logical>> {
    niri.pointer_visibility.is_visible().then(|| {
        niri.tablet_cursor_location
            .unwrap_or_else(|| niri.seat.get_pointer().unwrap().current_location())
    })
}

pub(super) struct MotionRedraw {
    // Ordinary motion touches few outputs; large cursors and monitor walls can spill safely.
    outputs: SmallVec<[Output; 4]>,
    position: Option<Point<f64, Logical>>,
    image: CursorImageStatus,
    focus: Option<MappedId>,
    active_output: Option<Output>,
    layer_focus: Option<LayerSurface>,
    all_outputs: bool,
}

impl MotionRedraw {
    pub fn new(niri: &Niri, position: Option<Point<f64, Logical>>) -> Self {
        let all_outputs = needs_all_outputs(niri);
        let mut outputs = SmallVec::new();
        if !all_outputs {
            add_overlapping_outputs(niri, position, &mut outputs);
        }
        Self {
            outputs,
            position,
            image: niri.cursor_manager.cursor_image().clone(),
            focus: niri.layout.focus().map(|window| window.id()),
            active_output: niri.layout.active_output().cloned(),
            layer_focus: niri.layer_shell_on_demand_focus.clone(),
            all_outputs,
        }
    }

    pub fn queue(mut self, niri: &mut Niri) {
        // Grabs, DnD, and compositor UIs can move or change content on other outputs.
        if self.all_outputs
            || needs_all_outputs(niri)
            || self.layer_focus != niri.layer_shell_on_demand_focus
        {
            niri.queue_redraw_all();
            return;
        }
        let position = displayed_position(niri);
        let focus_changed = self.focus != niri.layout.focus().map(|window| window.id())
            || self.active_output.as_ref() != niri.layout.active_output();
        if self.position == position
            && self.image == *niri.cursor_manager.cursor_image()
            && !focus_changed
        {
            return;
        }
        add_overlapping_outputs(niri, position, &mut self.outputs);
        // Focus-follows-mouse changes activation decoration and can scroll another output.
        if focus_changed {
            for output in self.active_output.iter().chain(niri.layout.active_output()) {
                if !self.outputs.contains(output) {
                    self.outputs.push(output.clone());
                }
            }
        }
        for output in self.outputs {
            if niri.output_state.contains_key(&output) {
                niri.queue_redraw(&output);
            }
        }
    }
}

pub(super) fn needs_all_outputs(niri: &Niri) -> bool {
    niri.seat.get_pointer().unwrap().is_grabbed()
        || niri.dnd_icon.is_some()
        || niri.screenshot_ui.is_open()
        || niri.window_mru_ui.is_open()
        || niri.layout.is_overview_open()
}

fn add_overlapping_outputs(
    niri: &Niri,
    position: Option<Point<f64, Logical>>,
    outputs: &mut SmallVec<[Output; 4]>,
) {
    let Some(position) = position else {
        return;
    };
    for output in niri.global_space.outputs() {
        // An output touched at the old location already needs a redraw, regardless of the
        // new position or cursor image. Avoid computing its bounds a second time.
        if outputs.contains(output) {
            continue;
        }
        let scale = output.current_scale();
        let bounds = match niri.cursor_manager.get_render_cursor(scale.integer_scale()) {
            RenderCursor::Hidden => continue,
            RenderCursor::Surface { hotspot, surface } => {
                let mut bounds = bbox_from_surface_tree(&surface, (0, 0)).to_f64();
                bounds.loc += position - hotspot.to_f64();
                bounds
            }
            RenderCursor::Named { scale, cursor, .. } => {
                let Some(mut bounds) = cursor.bounds(f64::from(scale)) else {
                    continue;
                };
                bounds.loc += position;
                bounds
            }
        };
        let geometry = niri.global_space.output_geometry(output).unwrap().to_f64();
        if overlaps_with_rounding(bounds, geometry, scale.fractional_scale()) {
            outputs.push(output.clone());
        }
    }
}

fn overlaps_with_rounding(
    mut image: Rectangle<f64, Logical>,
    output: Rectangle<f64, Logical>,
    scale: f64,
) -> bool {
    // Renderer positions are rounded to physical pixels independently for each output.
    let margin = 1. / scale;
    image.loc -= Point::from((margin, margin));
    image.size += Size::from((2. * margin, 2. * margin));
    image.overlaps(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state(
        output_count: u8,
    ) -> (
        calloop::EventLoop<'static, crate::niri::State>,
        crate::niri::State,
        Vec<Output>,
    ) {
        use crate::niri::State;

        let event_loop = calloop::EventLoop::try_new().unwrap();
        let display = smithay::reexports::wayland_server::Display::new().unwrap();
        let mut state = State::new(
            niri_config::Config::default(),
            event_loop.handle(),
            event_loop.get_signal(),
            display,
            true,
            false,
            false,
        )
        .unwrap();
        for number in 1..=output_count {
            state
                .backend
                .headless()
                .add_output(&mut state.niri, number, (800, 600));
        }
        let mut outputs: Vec<_> = state.niri.global_space.outputs().cloned().collect();
        outputs.sort_by_key(|output| output.name());
        (event_loop, state, outputs)
    }

    #[test]
    #[ignore = "manual CPU microbenchmark; run with --ignored --nocapture"]
    fn motion_redraw_microbenchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        use smithay::input::pointer::CursorIcon;

        use crate::niri::State;

        let event_loop = calloop::EventLoop::try_new().unwrap();
        let display = smithay::reexports::wayland_server::Display::new().unwrap();
        let mut config = niri_config::Config::default();
        config.cursor.xcursor_theme = "breeze_cursors".into();
        let mut state = State::new(
            config,
            event_loop.handle(),
            event_loop.get_signal(),
            display,
            true,
            false,
            false,
        )
        .unwrap();
        for number in 1..=3 {
            state
                .backend
                .headless()
                .add_output(&mut state.niri, number, (800, 600));
        }

        // Change the displayed location directly to time just the production damage
        // bookkeeping, excluding input dispatch, hit testing, and rendering.
        const ITERATIONS: u32 = 50_000;
        for icon in [CursorIcon::Default, CursorIcon::Wait] {
            state
                .niri
                .cursor_manager
                .set_cursor_image(CursorImageStatus::Named(icon));
            let RenderCursor::Named { cursor, .. } = state.niri.cursor_manager.get_render_cursor(1)
            else {
                unreachable!();
            };
            let mut samples = Vec::new();
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..ITERATIONS {
                    state.niri.tablet_cursor_location = Some((200., 200.).into());
                    let redraw = MotionRedraw::new(&state.niri, displayed_position(&state.niri));
                    state.niri.tablet_cursor_location = Some((201., 200.).into());
                    black_box(redraw).queue(black_box(&mut state.niri));
                }
                samples.push(start.elapsed().as_nanos() / u128::from(ITERATIONS));
            }
            samples.sort_unstable();
            eprintln!(
                "motion-redraw {icon:?} frames={} median={} ns/event samples={samples:?}",
                cursor.frames().len(),
                samples[3]
            );
        }
    }

    #[test]
    fn ordinary_motion_queues_old_and_new_outputs_but_overview_queues_all() {
        use crate::niri::RedrawState;

        let (_event_loop, mut state, outputs) = test_state(3);
        let position = |niri: &Niri, output: &Output| {
            niri.global_space
                .output_geometry(output)
                .unwrap()
                .loc
                .to_f64()
                + Point::from((200., 200.))
        };
        state.move_cursor(position(&state.niri, &outputs[0]));
        let redraw = MotionRedraw::new(&state.niri, displayed_position(&state.niri));
        assert!(!redraw.outputs.spilled());
        state.move_cursor(position(&state.niri, &outputs[1]));
        for output in state.niri.output_state.values_mut() {
            output.redraw_state = RedrawState::Idle;
        }
        redraw.queue(&mut state.niri);
        assert!(state.niri.is_queued(&outputs[0]));
        assert!(state.niri.is_queued(&outputs[1]));
        assert!(!state.niri.is_queued(&outputs[2]));

        state.niri.layout.toggle_overview();
        let redraw = MotionRedraw::new(&state.niri, displayed_position(&state.niri));
        for output in state.niri.output_state.values_mut() {
            output.redraw_state = RedrawState::Idle;
        }
        redraw.queue(&mut state.niri);
        assert!(outputs.iter().all(|output| state.niri.is_queued(output)));
    }

    #[test]
    fn large_overlap_spills_without_losing_or_duplicating_outputs() {
        use crate::niri::RedrawState;

        let (_event_loop, mut state, outputs) = test_state(8);
        // Mirrored outputs all intersect the cursor, exceeding the inline capacity.
        for output in &outputs {
            state.niri.global_space.map_output(output, (0, 0));
        }
        state.move_cursor((200., 200.).into());
        let mut redraw = MotionRedraw::new(&state.niri, displayed_position(&state.niri));
        assert!(redraw.outputs.spilled());
        assert_eq!(redraw.outputs.len(), outputs.len());
        add_overlapping_outputs(&state.niri, Some((201., 200.).into()), &mut redraw.outputs);
        assert_eq!(redraw.outputs.len(), outputs.len());
        state.move_cursor((201., 200.).into());
        for output in state.niri.output_state.values_mut() {
            output.redraw_state = RedrawState::Idle;
        }
        redraw.queue(&mut state.niri);
        assert!(outputs.iter().all(|output| state.niri.is_queued(output)));
    }

    #[test]
    fn cursor_image_changes_redraw_without_pointer_motion() {
        use crate::niri::RedrawState;

        let (_event_loop, mut state, outputs) = test_state(2);
        let position = state
            .niri
            .global_space
            .output_geometry(&outputs[0])
            .unwrap()
            .loc
            .to_f64()
            + Point::from((200., 200.));
        state.move_cursor(position);
        for image in [
            CursorImageStatus::Hidden,
            CursorImageStatus::default_named(),
        ] {
            let redraw = MotionRedraw::new(&state.niri, displayed_position(&state.niri));
            state.niri.cursor_manager.set_cursor_image(image);
            for output in state.niri.output_state.values_mut() {
                output.redraw_state = RedrawState::Idle;
            }
            redraw.queue(&mut state.niri);
            assert!(state.niri.is_queued(&outputs[0]));
            assert!(!state.niri.is_queued(&outputs[1]));
        }
    }

    #[test]
    fn cursor_overlaps_adjacent_output_beyond_its_hotspot() {
        let image = Rectangle::new((95., 50.).into(), (20., 20.).into());
        let outputs = [0., 100., 200.].map(|x| Rectangle::new((x, 0.).into(), (100., 100.).into()));
        assert!(overlaps_with_rounding(image, outputs[0], 1.));
        assert!(overlaps_with_rounding(image, outputs[1], 2.));
        assert!(!overlaps_with_rounding(image, outputs[2], 1.25));
    }

    #[test]
    fn physical_rounding_cannot_leave_a_stale_edge_pixel() {
        let image = Rectangle::new((90., 20.).into(), (9.8, 10.).into());
        let output = Rectangle::new((100., 0.).into(), (100., 100.).into());
        assert!(overlaps_with_rounding(image, output, 2.));
    }
}
