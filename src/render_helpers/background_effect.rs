use std::sync::{Arc, Mutex};

use niri_config::CornerRadius;
use smithay::backend::renderer::element::RenderElement;
use smithay::backend::renderer::vulkan::{VulkanError, VulkanFrame, VulkanRenderer};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale};
use smithay::wayland::compositor::{with_states, SurfaceData};
use wayland_server::protocol::wl_surface::WlSurface;

use super::renderer::NiriRenderer;
use crate::handlers::background_effect::get_cached_blur_region;
use crate::niri_render_elements;
use crate::render_helpers::blur::BlurOptions;
use crate::render_helpers::damage::ExtraDamage;
use crate::render_helpers::framebuffer_effect::{FramebufferEffect, FramebufferEffectElement};
use crate::render_helpers::xray::{XrayElement, XrayPos};
use crate::render_helpers::RenderCtx;
use crate::utils::region::TransformedRegion;
use crate::utils::surface_geo;

#[derive(Debug)]
pub struct BackgroundEffect {
    nonxray: FramebufferEffect,
    /// Damage when options change.
    damage: ExtraDamage,
    /// Corner radius for clipping.
    ///
    /// Stored here in addition to `RenderParams` to damage when it changes.
    // FIXME: would be good to remove this duplication of radius.
    corner_radius: CornerRadius,
    blur_config: niri_config::Blur,
    options: Options,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Options {
    pub blur: bool,
    pub xray: bool,
    pub noise: Option<f64>,
    pub saturation: Option<f64>,
}

impl Options {
    fn is_visible(&self) -> bool {
        self.xray
            || self.blur
            || self.noise.is_some_and(|x| x > 0.)
            || self.saturation.is_some_and(|x| x != 1.)
    }
}

/// Render-time parameters.
#[derive(Debug)]
pub struct RenderParams {
    /// Geometry of the background effect.
    pub geometry: Rectangle<f64, Logical>,
    /// Effect subregion, will be clipped to `geometry`.
    ///
    /// `subregion.iter()` should return `geometry`-relative rectangles.
    pub subregion: Option<TransformedRegion>,
    /// Geometry and radius for clipping in the same coordinate space as `geometry`.
    pub clip: Option<(Rectangle<f64, Logical>, CornerRadius)>,
    /// Scale to use for rounding to physical pixels.
    pub scale: f64,
}

impl RenderParams {
    fn fit_clip_radius(&mut self) {
        if let Some((geo, radius)) = &mut self.clip {
            // HACK: increase radius to avoid slight bleed on rounded corners.
            *radius = radius.expanded_by(1.);

            *radius = radius.fit_to(geo.size.w as f32, geo.size.h as f32);
        }
    }

    fn is_covered_by(&self, opaque: Rectangle<f64, Logical>) -> bool {
        // Elements round their position and size separately. Logical containment alone does not
        // guarantee that the effect stays inside the backdrop after rounding to physical pixels.
        opaque.contains_rect(self.geometry)
            && opaque
                .to_physical_precise_round::<_, i32>(self.scale)
                .contains_rect(self.geometry.to_physical_precise_round(self.scale))
    }
}

niri_render_elements! {
    BackgroundEffectElement => {
        FramebufferEffect = FramebufferEffectElement,
        Xray = XrayElement,
        ExtraDamage = ExtraDamage,
    }
}

impl RenderElement<VulkanRenderer> for BackgroundEffectElement {
    fn capture_framebuffer(
        &self,
        frame: &mut VulkanFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        cache: &UserDataMap,
    ) -> Result<(), VulkanError> {
        match self {
            BackgroundEffectElement::FramebufferEffect(elem) => {
                RenderElement::<VulkanRenderer>::capture_framebuffer(elem, frame, src, dst, cache)
            }
            BackgroundEffectElement::Xray(elem) => {
                RenderElement::<VulkanRenderer>::capture_framebuffer(elem, frame, src, dst, cache)
            }
            BackgroundEffectElement::ExtraDamage(elem) => {
                RenderElement::<VulkanRenderer>::capture_framebuffer(elem, frame, src, dst, cache)
            }
        }
    }

    fn draw(
        &self,
        frame: &mut VulkanFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), VulkanError> {
        match self {
            BackgroundEffectElement::FramebufferEffect(elem) => {
                RenderElement::<VulkanRenderer>::draw(
                    elem,
                    frame,
                    src,
                    dst,
                    damage,
                    opaque_regions,
                    cache,
                )
            }
            BackgroundEffectElement::Xray(elem) => RenderElement::<VulkanRenderer>::draw(
                elem,
                frame,
                src,
                dst,
                damage,
                opaque_regions,
                cache,
            ),
            BackgroundEffectElement::ExtraDamage(elem) => RenderElement::<VulkanRenderer>::draw(
                elem,
                frame,
                src,
                dst,
                damage,
                opaque_regions,
                cache,
            ),
        }
    }
}

impl BackgroundEffect {
    pub fn new() -> Self {
        Self {
            nonxray: FramebufferEffect::new(),
            damage: ExtraDamage::new(),
            corner_radius: CornerRadius::default(),
            blur_config: niri_config::Blur::default(),
            options: Options::default(),
        }
    }

    /// Damage the background effect, for example when a blur subregion changes.
    pub fn damage(&mut self) {
        self.damage.damage_all();
        self.nonxray.damage();
    }

    pub fn update_config(&mut self, config: niri_config::Blur) {
        if self.blur_config == config {
            return;
        }

        self.blur_config = config;
        self.damage.damage_all();
        self.nonxray.damage();
    }

    pub fn update_render_elements(
        &mut self,
        corner_radius: CornerRadius,
        effect: niri_config::BackgroundEffect,
        has_blur_region: bool,
    ) {
        // If the surface explicitly requests a blur region, default blur to true.
        let blur = if has_blur_region {
            effect.blur != Some(false)
        } else {
            effect.blur == Some(true)
        };

        let mut options = Options {
            blur,
            xray: effect.xray == Some(true),
            noise: effect.noise,
            saturation: effect.saturation,
        };

        // If we have some background effect but xray wasn't explicitly set, default it to true
        // since it's cheaper.
        if options.is_visible() && effect.xray.is_none() {
            options.xray = true;
        }

        if self.options == options && self.corner_radius == corner_radius {
            return;
        }

        self.options = options;
        self.corner_radius = corner_radius;
        self.damage.damage_all();
        self.nonxray.damage();
    }

    pub fn is_visible(&self) -> bool {
        self.options.is_visible()
    }

    pub fn render<R: NiriRenderer>(
        &self,
        ctx: RenderCtx<R>,
        ns: Option<usize>,
        mut params: RenderParams,
        xray_pos: XrayPos,
        push: &mut dyn FnMut(BackgroundEffectElement),
    ) {
        if !self.is_visible() {
            return;
        }

        if let Some(clip) = &mut params.clip {
            clip.1 = self.corner_radius;
        }
        params.fit_clip_radius();

        let damage = self.damage.render(params.geometry);

        // Use noise/saturation from options, falling back to blur defaults if blurred, and
        // to no effect if not blurred.
        let blur = self.options.blur && !self.blur_config.off;
        let blur_options = blur.then_some(BlurOptions::from(self.blur_config));
        let noise = if blur { self.blur_config.noise } else { 0. };
        let noise = self.options.noise.unwrap_or(noise) as f32;
        let saturation = if blur {
            self.blur_config.saturation
        } else {
            1.
        };
        let saturation = self.options.saturation.unwrap_or(saturation) as f32;

        if self.options.xray {
            let Some(xray) = ctx.xray else {
                return;
            };

            push(damage.into());
            xray.render(
                ctx,
                params,
                xray_pos,
                blur,
                noise,
                saturation,
                &mut |elem| push(elem.into()),
            );
        } else {
            // Render non-xray effect.
            let elem = self
                .nonxray
                .render(ns, params, blur_options, noise, saturation);
            push(elem.into());
        }
    }
}

fn render_params_for_tile(
    geometry: Rectangle<f64, Logical>,
    scale: f64,
    clip_to_geometry: bool,
    block_out: bool,
    blur_region: Option<Arc<Vec<Rectangle<i32, Logical>>>>,
    surface_geo: Rectangle<f64, Logical>,
    surface_anim_scale: Scale<f64>,
) -> Option<RenderParams> {
    // Effects not requested by the surface itself are drawn to match the geometry.
    let mut clip = true;

    let mut effect_geometry = geometry;
    let mut subregion = None;
    if let Some(rects) = blur_region {
        if rects.is_empty() {
            // Surface has a set, but empty blur region.
            return None;
        } else {
            // If the surface itself requests the effects, apply different defaults.
            clip = clip_to_geometry;

            // Use geometry-shaped blur for blocked-out windows to avoid unintentionally
            // leaking any surface shapes. We render those windows as geometry-shaped solid
            // rectangles anyway.
            if block_out {
                clip = true;
            } else {
                let mut surface_geo = surface_geo.upscale(surface_anim_scale);
                surface_geo.loc += geometry.loc;

                subregion = Some(TransformedRegion {
                    rects,
                    scale: surface_anim_scale,
                    offset: surface_geo.loc,
                });

                surface_geo = surface_geo
                    .to_physical_precise_round(scale)
                    .to_logical(scale);
                effect_geometry = surface_geo;
            }
        }
    }

    // This corner radius is reset to self.corner_radius in render().
    let clip = clip.then_some((geometry, CornerRadius::default()));

    Some(RenderParams {
        geometry: effect_geometry,
        subregion,
        clip,
        scale,
    })
}

/// Per-surface background effect stored in its data map.
struct SurfaceBackgroundEffect(Mutex<BackgroundEffect>);

impl SurfaceBackgroundEffect {
    fn get(states: &SurfaceData) -> &Self {
        states
            .data_map
            .get_or_insert(|| SurfaceBackgroundEffect(Mutex::new(BackgroundEffect::new())))
    }
}

pub fn damage_surface(states: &SurfaceData) {
    if let Some(effect) = states.data_map.get::<SurfaceBackgroundEffect>() {
        effect.0.lock().unwrap().damage();
    }
}

// Silence, Clippy
// A Smithay user is talking
#[allow(clippy::too_many_arguments)]
pub fn render_for_tile<R: NiriRenderer>(
    ctx: RenderCtx<R>,
    ns: Option<usize>,
    geometry: Rectangle<f64, Logical>,
    opaque_region: Option<Rectangle<f64, Logical>>,
    scale: f64,
    clip_to_geometry: bool,
    surface: &WlSurface,
    surface_off: Point<f64, Logical>,
    surface_anim_scale: Scale<f64>,
    blur_config: niri_config::Blur,
    radius: CornerRadius,
    effect: niri_config::BackgroundEffect,
    should_block_out: bool,
    xray_pos: XrayPos,
    push: &mut dyn FnMut(BackgroundEffectElement),
) {
    with_states(surface, |states| {
        let background_effect = SurfaceBackgroundEffect::get(states);
        let mut background_effect = background_effect.0.lock().unwrap();

        let blur_region = get_cached_blur_region(states);
        let has_blur_region = blur_region.as_ref().is_some_and(|r| !r.is_empty());

        background_effect.update_config(blur_config);
        background_effect.update_render_elements(radius, effect, has_blur_region);

        if !background_effect.is_visible() {
            return;
        }

        let mut surface_geo = surface_geo(states).unwrap_or_default().to_f64();
        surface_geo.loc += surface_off;

        let Some(params) = render_params_for_tile(
            geometry,
            scale,
            clip_to_geometry,
            should_block_out,
            blur_region,
            surface_geo,
            surface_anim_scale,
        ) else {
            return;
        };

        // Skip before preparing xray's offscreen contents. Client blur regions can extend past
        // the window geometry, so check the final effect geometry against the opaque backdrop.
        // Overview scaling introduces another rounding step; keep that path unchanged.
        if xray_pos.zoom == 1. && opaque_region.is_some_and(|opaque| params.is_covered_by(opaque)) {
            return;
        }

        let xray_pos = xray_pos.offset(params.geometry.loc - geometry.loc);
        background_effect.render(ctx, ns, params, xray_pos, push);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_backdrop_covers_effect_at_fractional_scales() {
        let geometry = Rectangle::new((10., 20.).into(), (800., 600.).into());
        for scale in [1., 1.25, 1.5, 2.] {
            let params = render_params_for_tile(
                geometry,
                scale,
                false,
                false,
                None,
                geometry,
                Scale::from(1.),
            )
            .unwrap();
            assert!(params.is_covered_by(geometry));
            assert!(params.is_covered_by(Rectangle::new((0., 0.).into(), (1920., 1080.).into(),)));
            assert!(!params.is_covered_by(Rectangle::new((11., 20.).into(), (800., 600.).into(),)));
        }
    }

    #[test]
    fn logical_coverage_must_survive_pixel_rounding() {
        // Both right edges are at 1.5 logically, but position and size round independently:
        // the backdrop ends at pixel 1 while the effect ends at pixel 2.
        let opaque = Rectangle::new((0.25, 0.).into(), (1.25, 10.).into());
        let geometry = Rectangle::new((0.5, 0.).into(), (1., 10.).into());
        let params =
            render_params_for_tile(geometry, 1., false, false, None, geometry, Scale::from(1.))
                .unwrap();
        assert!(opaque.contains_rect(geometry));
        assert!(!params.is_covered_by(opaque));
    }

    #[test]
    fn client_blur_region_outside_backdrop_is_not_covered() {
        let geometry = Rectangle::from_size((100., 100.).into());
        let surface_geo = Rectangle::new((-8., -8.).into(), (116., 116.).into());
        let blur_region = Arc::new(vec![Rectangle::from_size((116, 116).into())]);

        for block_out in [false, true] {
            let params = render_params_for_tile(
                geometry,
                1.25,
                false,
                block_out,
                Some(blur_region.clone()),
                surface_geo,
                Scale::from(1.),
            )
            .unwrap();
            // Blocked-out captures intentionally constrain effects to the window geometry.
            assert_eq!(params.is_covered_by(geometry), block_out);
        }
    }
}
