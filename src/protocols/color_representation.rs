use std::cell::RefCell;

use smithay::backend::allocator::{format::get_bpp, Buffer as _};
use smithay::backend::renderer::utils::RendererSurfaceStateUserData;
use smithay::backend::renderer::{buffer_type, BufferType};
use smithay::reexports::wayland_protocols::wp::color_representation::v1::server::{
    wp_color_representation_manager_v1::{self as manager, WpColorRepresentationManagerV1},
    wp_color_representation_surface_v1::{
        self as representation, AlphaMode, Coefficients, Range, WpColorRepresentationSurfaceV1,
    },
};
use smithay::reexports::wayland_server::backend::ClientId;
use smithay::reexports::wayland_server::protocol::{wl_buffer::WlBuffer, wl_surface::WlSurface};
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, WEnum, Weak,
};
use smithay::wayland::compositor::{
    add_post_commit_hook, with_states, BufferAssignment, Cacheable, SurfaceAttributes,
};
use smithay::wayland::dmabuf::get_dmabuf;

use super::EmptyData;
use crate::niri::State;

type SurfaceBinding = RefCell<Option<WpColorRepresentationSurfaceV1>>;

#[derive(Clone, Copy, Default)]
struct SurfaceRepresentation {
    identity: bool,
    has_chroma_location: bool,
}

impl Cacheable for SurfaceRepresentation {
    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        *self
    }

    fn merge_into(self, current: &mut Self, _dh: &DisplayHandle) {
        *current = self;
    }
}

pub(crate) fn create_global(display: &DisplayHandle) {
    display.create_global::<State, WpColorRepresentationManagerV1, _>(1, EmptyData);
}

impl GlobalDispatch<WpColorRepresentationManagerV1, EmptyData> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpColorRepresentationManagerV1>,
        _data: &EmptyData,
        data_init: &mut DataInit<Self>,
    ) {
        let manager = data_init.init(resource, EmptyData);
        // The renderer already consumes full-range RGB with electrical premultiplication.
        manager.supported_alpha_mode(AlphaMode::PremultipliedElectrical);
        manager.supported_coefficients_and_ranges(Coefficients::Identity, Range::Full);
        manager.done();
    }
}

impl Dispatch<WpColorRepresentationManagerV1, EmptyData> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &WpColorRepresentationManagerV1,
        request: manager::Request,
        _data: &EmptyData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<Self>,
    ) {
        let manager::Request::GetSurface { id, surface } = request else {
            return;
        };
        let first_binding = with_states(&surface, |states| {
            let first_binding = states.data_map.insert_if_missing(SurfaceBinding::default);
            let mut binding = states
                .data_map
                .get::<SurfaceBinding>()
                .unwrap()
                .borrow_mut();
            if binding.is_some() {
                resource.post_error(
                    manager::Error::SurfaceExists,
                    "surface already has color representation",
                );
            } else {
                *binding = Some(data_init.init(id, surface.downgrade()));
            }
            first_binding
        });
        if first_binding {
            add_post_commit_hook::<Self, _>(&surface, |_, _, surface| validate_buffer(surface));
        }
    }
}

impl Dispatch<WpColorRepresentationSurfaceV1, Weak<WlSurface>> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &WpColorRepresentationSurfaceV1,
        request: representation::Request,
        surface: &Weak<WlSurface>,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<Self>,
    ) {
        use representation::{Error, Request};
        if matches!(request, Request::Destroy) {
            return;
        }
        let Ok(surface) = surface.upgrade() else {
            resource.post_error(Error::Inert, "the underlying wl_surface was destroyed");
            return;
        };

        match request {
            Request::SetAlphaMode {
                alpha_mode: WEnum::Value(AlphaMode::PremultipliedElectrical),
            } => (),
            Request::SetAlphaMode { .. } => {
                resource.post_error(Error::AlphaMode, "unsupported alpha mode")
            }
            Request::SetCoefficientsAndRange {
                coefficients: WEnum::Value(Coefficients::Identity),
                range: WEnum::Value(Range::Full),
            } => with_states(&surface, |states| {
                states
                    .cached_state
                    .get::<SurfaceRepresentation>()
                    .pending()
                    .identity = true;
            }),
            Request::SetCoefficientsAndRange { .. } => {
                resource.post_error(Error::Coefficients, "only full-range RGB is supported");
            }
            Request::SetChromaLocation {
                chroma_location: WEnum::Value(_),
            } => with_states(&surface, |states| {
                states
                    .cached_state
                    .get::<SurfaceRepresentation>()
                    .pending()
                    .has_chroma_location = true;
            }),
            Request::SetChromaLocation { .. } => {
                resource.post_error(Error::ChromaLocation, "invalid chroma location");
            }
            _ => (),
        }
    }

    fn destroyed(
        _state: &mut Self,
        _client: ClientId,
        _resource: &WpColorRepresentationSurfaceV1,
        surface: &Weak<WlSurface>,
    ) {
        if let Ok(surface) = surface.upgrade() {
            with_states(&surface, |states| {
                states.data_map.get::<SurfaceBinding>().unwrap().take();
                *states.cached_state.get::<SurfaceRepresentation>().pending() =
                    SurfaceRepresentation::default();
            });
        }
    }
}

fn validate_buffer(surface: &WlSurface) {
    let error_resource = with_states(surface, |states| {
        let binding = states.data_map.get::<SurfaceBinding>().unwrap().borrow();
        let binding = binding.as_ref()?;
        let representation = *states.cached_state.get::<SurfaceRepresentation>().current();
        if !representation.has_chroma_location && !representation.identity {
            return None;
        }

        // RGB has no subsampled chroma. Check the committed attachment, including commits
        // that change the representation while leaving the previous buffer in place.
        let incompatible =
            |buffer: &WlBuffer| representation.has_chroma_location || !is_rgb(buffer);
        let incompatible = match &states
            .cached_state
            .get::<SurfaceAttributes>()
            .current()
            .buffer
        {
            Some(BufferAssignment::NewBuffer(buffer)) => incompatible(buffer),
            Some(BufferAssignment::Removed) => false,
            None => states
                .data_map
                .get::<RendererSurfaceStateUserData>()
                .is_some_and(|renderer| {
                    renderer
                        .lock()
                        .unwrap()
                        .buffer()
                        .is_some_and(|buffer| incompatible(buffer))
                }),
        };
        incompatible.then(|| binding.clone())
    });

    if let Some(resource) = error_resource {
        resource.post_error(
            representation::Error::PixelFormat,
            "buffer must use full-range RGB without a chroma location",
        );
    }
}

fn is_rgb(buffer: &WlBuffer) -> bool {
    if let Ok(dmabuf) = get_dmabuf(buffer) {
        // Smithay defines the RGB layouts supported by its renderers in this table.
        get_bpp(dmabuf.format().code).is_some()
    } else {
        matches!(
            buffer_type(buffer),
            Some(BufferType::Shm | BufferType::SinglePixel)
        )
    }
}
