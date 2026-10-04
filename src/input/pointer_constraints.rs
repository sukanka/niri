//! Motion clipping for the intersection of a surface's input and confinement regions.

use smithay::backend::renderer::utils::with_renderer_surface_state;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point, Rectangle};
use smithay::wayland::compositor::{
    with_states, RectangleKind, RegionAttributes, SurfaceAttributes,
};

pub(super) fn confine_motion(
    surface: &WlSurface,
    surface_origin: Point<f64, Logical>,
    region: Option<&RegionAttributes>,
    from: Point<f64, Logical>,
    to: Point<f64, Logical>,
) -> Point<f64, Logical> {
    let Some(size) = with_renderer_surface_state(surface, |state| state.surface_size()).flatten()
    else {
        return from;
    };
    let input_region = with_states(surface, |states| {
        states
            .cached_state
            .get::<SurfaceAttributes>()
            .current()
            .input_region
            .clone()
    });
    let mut rects = effective_region(
        Rectangle::from_size(size).to_f64(),
        input_region.as_ref(),
        region,
    );
    // Clip in global coordinates: translating an already-clipped local point afterwards can
    // round its exclusive edge back out of the region when the surface has a large offset.
    for rect in &mut rects {
        rect.loc += surface_origin;
    }
    clip_motion(&rects, from, to)
}

fn region_rectangles(
    bounds: Rectangle<f64, Logical>,
    region: Option<&RegionAttributes>,
) -> Vec<Rectangle<f64, Logical>> {
    let Some(region) = region else {
        return vec![bounds];
    };
    let mut rectangles = Vec::new();
    for (kind, rectangle) in &region.rects {
        // Convert before adding coordinates, so even client rectangles near i32::MAX cannot
        // overflow while clipping them to the surface.
        let Some(rectangle) = bounds.intersection(rectangle.to_f64()) else {
            continue;
        };
        match kind {
            RectangleKind::Add => rectangles.push(rectangle),
            RectangleKind::Subtract => {
                rectangles = Rectangle::subtract_rects_many_in_place(rectangles, [rectangle]);
            }
        }
    }
    rectangles
}

fn effective_region(
    bounds: Rectangle<f64, Logical>,
    input: Option<&RegionAttributes>,
    confine: Option<&RegionAttributes>,
) -> Vec<Rectangle<f64, Logical>> {
    let input = region_rectangles(bounds, input);
    let confine = region_rectangles(bounds, confine);
    input
        .iter()
        .flat_map(|a| confine.iter().filter_map(|b| a.intersection(*b)))
        .collect()
}

fn contains(rects: &[Rectangle<f64, Logical>], point: Point<f64, Logical>) -> bool {
    rects.iter().any(|rect| rect.contains(point))
}

/// Clip along the segment, never jumping across a hole or to a disconnected rectangle.
fn clip_segment(
    rects: &[Rectangle<f64, Logical>],
    from: Point<f64, Logical>,
    to: Point<f64, Logical>,
) -> Point<f64, Logical> {
    let delta = to - from;
    let mut intervals = Vec::with_capacity(rects.len());
    for rect in rects {
        let mut enter: f64 = 0.;
        let mut leave: f64 = 1.;
        for (origin, movement, low, high) in [
            (from.x, delta.x, rect.loc.x, rect.loc.x + rect.size.w),
            (from.y, delta.y, rect.loc.y, rect.loc.y + rect.size.h),
        ] {
            if movement == 0. {
                if origin < low || origin >= high {
                    leave = -1.;
                    break;
                }
            } else {
                let a = (low - origin) / movement;
                let b = (high - origin) / movement;
                enter = enter.max(a.min(b));
                leave = leave.min(a.max(b));
            }
        }
        if enter <= leave {
            intervals.push((enter, leave));
        }
    }
    intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut end: f64 = 0.;
    for (enter, leave) in intervals {
        if enter > end {
            break;
        }
        end = end.max(leave);
    }
    if end >= 1. && contains(rects, to) {
        return to;
    }
    let mut result = from + Point::from((delta.x * end, delta.y * end));
    // Right and bottom edges are exclusive. Move a few ULPs back along the segment, rather
    // than subtracting a fixed pixel or permitting a tolerance that can bridge narrow holes.
    for _ in 0..4 {
        if contains(rects, result) {
            return result;
        }
        if delta.x > 0. {
            result.x = result.x.next_down();
        }
        if delta.x < 0. {
            result.x = result.x.next_up();
        }
        if delta.y > 0. {
            result.y = result.y.next_down();
        }
        if delta.y < 0. {
            result.y = result.y.next_up();
        }
    }
    from
}

fn clip_motion(
    rects: &[Rectangle<f64, Logical>],
    from: Point<f64, Logical>,
    to: Point<f64, Logical>,
) -> Point<f64, Logical> {
    if !contains(rects, from) || !to.x.is_finite() || !to.y.is_finite() {
        return from;
    }
    let hit = clip_segment(rects, from, to);
    if hit == to {
        return to;
    }
    // Preserve tangential movement after hitting an edge. Only one axis can slide; do not
    // reinstate the blocked component after moving around the corner of an obstacle.
    let x = clip_segment(rects, hit, Point::from((to.x, hit.y)));
    let y = clip_segment(rects, hit, Point::from((hit.x, to.y)));
    let distance = |point: Point<f64, Logical>| (point.x - to.x).hypot(point.y - to.y);
    if distance(x) < distance(y) {
        x
    } else {
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Logical> {
        Rectangle::new((x, y).into(), (w, h).into())
    }

    fn clip(
        region: &[Rectangle<f64, Logical>],
        from: (f64, f64),
        to: (f64, f64),
    ) -> Point<f64, Logical> {
        clip_motion(region, from.into(), to.into())
    }

    #[test]
    fn slides_along_exclusive_surface_edge() {
        let region = [rect(0, 0, 100, 100).to_f64()];
        let point = clip(&region, (50., 50.), (200., 80.));
        assert!(point.x < 100. && point.x > 99.99);
        assert_eq!(point.y, 80.);
        assert!(contains(&region, point));
        let point = clip_motion(&region, point, (120., 20.).into());
        assert!(point.x < 100.);
        assert_eq!(point.y, 20.);
    }

    #[test]
    fn clips_in_surface_coordinates_with_nonzero_origin() {
        let origin = Point::from((1900., -800.));
        let from = Point::from((1950., -750.));
        let to = Point::from((2100., -720.));
        let region = [Rectangle::new(origin, (100., 100.).into())];
        let result = clip_motion(&region, from, to);
        assert!(result.x < 2000. && result.x > 1999.99);
        assert_eq!(result.y, -720.);
    }

    #[test]
    fn cannot_tunnel_between_disconnected_rectangles() {
        let region = [rect(0, 0, 10, 10).to_f64(), rect(20, 0, 10, 10).to_f64()];
        let point = clip(&region, (5., 5.), (25., 5.));
        assert!(point.x < 10.);
        assert_eq!(point.y, 5.);
    }

    #[test]
    fn connected_rectangles_do_not_create_artificial_edges() {
        let region = [rect(0, 0, 10, 10).to_f64(), rect(10, 0, 10, 10).to_f64()];
        assert_eq!(clip(&region, (5., 5.), (15., 5.)), Point::from((15., 5.)));
    }

    #[test]
    fn subtraction_holes_and_later_additions_preserve_region_order() {
        let bounds = rect(0, 0, 100, 100).to_f64();
        let mut region = RegionAttributes {
            rects: vec![
                (RectangleKind::Add, rect(0, 0, 100, 100)),
                (RectangleKind::Subtract, rect(40, 0, 20, 100)),
            ],
        };
        let effective = effective_region(bounds, None, Some(&region));
        let result = clip(&effective, (20., 50.), (80., 50.));
        assert!(result.x < 40.);
        region
            .rects
            .push((RectangleKind::Add, rect(40, 45, 20, 10)));
        let effective = effective_region(bounds, None, Some(&region));
        assert_eq!(
            clip(&effective, (20., 50.), (80., 50.)),
            Point::from((80., 50.))
        );
    }

    #[test]
    fn constraint_is_intersected_with_surface_input_region() {
        let input = RegionAttributes {
            rects: vec![(RectangleKind::Add, rect(10, 20, 40, 30))],
        };
        let confine = RegionAttributes {
            rects: vec![(RectangleKind::Add, rect(-10, -10, 500, 500))],
        };
        let effective =
            effective_region(rect(0, 0, 100, 100).to_f64(), Some(&input), Some(&confine));
        let result = clip(&effective, (20., 30.), (80., 40.));
        assert!(result.x < 50. && result.x > 49.99);
        assert_eq!(result.y, 40.);
        assert_eq!(clip(&effective, (20., 30.), (-500., 40.)).x, 10.);
    }

    #[test]
    fn large_motion_does_not_bridge_one_pixel_hole() {
        let region = [
            rect(0, 0, 10, 10).to_f64(),
            rect(11, 0, i32::MAX - 11, 10).to_f64(),
        ];
        let result = clip(&region, (5., 5.), (1e15, 5.));
        assert!(result.x < 10.);
    }

    #[test]
    fn empty_regions_and_nonfinite_events_do_not_move_pointer() {
        let region = [rect(0, 0, 10, 10).to_f64()];
        assert_eq!(clip(&[], (5., 5.), (7., 7.)), Point::from((5., 5.)));
        assert_eq!(
            clip(&region, (5., 5.), (f64::NAN, 7.)),
            Point::from((5., 5.))
        );
    }
}
