//! Output damage caused by ordinary pointer motion.

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
    outputs: Vec<Output>,
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
        Self {
            outputs: if all_outputs {
                Vec::new()
            } else {
                overlapping_outputs(niri, position)
            },
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
        self.outputs.extend(overlapping_outputs(niri, position));
        // Focus-follows-mouse changes activation decoration and can scroll another output.
        if focus_changed {
            self.outputs.extend(self.active_output);
            self.outputs.extend(niri.layout.active_output().cloned());
        }
        let mut queued = Vec::new();
        for output in self.outputs {
            if !queued.contains(&output) && niri.output_state.contains_key(&output) {
                niri.queue_redraw(&output);
                queued.push(output);
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

fn overlapping_outputs(niri: &Niri, position: Option<Point<f64, Logical>>) -> Vec<Output> {
    let Some(position) = position else {
        return Vec::new();
    };
    niri.global_space
        .outputs()
        .filter(|output| {
            let scale = output.current_scale();
            let bounds = match niri.cursor_manager.get_render_cursor(scale.integer_scale()) {
                RenderCursor::Hidden => return false,
                RenderCursor::Surface { hotspot, surface } => {
                    let mut bounds = bbox_from_surface_tree(&surface, (0, 0)).to_f64();
                    bounds.loc += position - hotspot.to_f64();
                    bounds
                }
                RenderCursor::Named { scale, cursor, .. } => {
                    // Include all animation frames: the old displayed frame can have a different
                    // hotspot or size from the current frame, and both must be erased correctly.
                    let bounds = named_bounds(
                        cursor
                            .frames()
                            .iter()
                            .map(|frame| (frame.width, frame.height, frame.xhot, frame.yhot)),
                        f64::from(scale),
                    );
                    let Some(mut bounds) = bounds else {
                        return false;
                    };
                    bounds.loc += position;
                    bounds
                }
            };
            let geometry = niri.global_space.output_geometry(output).unwrap().to_f64();
            overlaps_with_rounding(bounds, geometry, scale.fractional_scale())
        })
        .cloned()
        .collect()
}

fn named_bounds(
    frames: impl Iterator<Item = (u32, u32, u32, u32)>,
    scale: f64,
) -> Option<Rectangle<f64, Logical>> {
    frames
        .map(|(width, height, xhot, yhot)| {
            Rectangle::new(
                Point::from((-f64::from(xhot) / scale, -f64::from(yhot) / scale)),
                Size::from((f64::from(width) / scale, f64::from(height) / scale)),
            )
        })
        .reduce(|a, b| a.merge(b))
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

    #[test]
    fn ordinary_motion_queues_old_and_new_outputs_but_overview_queues_all() {
        use crate::niri::{RedrawState, State};

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
        for number in 1..=3 {
            state
                .backend
                .headless()
                .add_output(&mut state.niri, number, (800, 600));
        }
        let mut outputs: Vec<_> = state.niri.global_space.outputs().cloned().collect();
        outputs.sort_by_key(|output| output.name());
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
    fn animated_cursor_bounds_include_old_hotspot_and_scale() {
        let bounds = named_bounds([(32, 32, 0, 0), (64, 48, 32, 16)].into_iter(), 2.).unwrap();
        assert_eq!(bounds.loc, Point::from((-16., -8.)));
        assert_eq!(bounds.size, Size::from((32., 24.)));
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
