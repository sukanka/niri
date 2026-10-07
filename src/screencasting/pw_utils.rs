use std::cell::{Cell, RefCell};
use std::cmp::min;
use std::collections::HashMap;
use std::io::Cursor;
use std::iter::zip;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
use std::ptr::NonNull;
use std::rc::Rc;
use std::time::{Duration, Instant};
use std::{mem, slice};

use anyhow::{bail, ensure, Context as _};
use calloop::timer::{TimeoutAction, Timer};
use calloop::RegistrationToken;
use pipewire::context::ContextRc;
use pipewire::core::{CoreRc, PW_ID_CORE};
use pipewire::loop_::Timeout;
use pipewire::main_loop::MainLoopRc;
use pipewire::properties::PropertiesBox;
use pipewire::spa::buffer::DataType;
use pipewire::spa::param::format::{FormatProperties, MediaSubtype, MediaType};
use pipewire::spa::param::format_utils::parse_format;
use pipewire::spa::param::video::{VideoFormat, VideoInfoRaw};
use pipewire::spa::param::ParamType;
use pipewire::spa::pod::deserialize::PodDeserializer;
use pipewire::spa::pod::serialize::PodSerializer;
use pipewire::spa::pod::{self, ChoiceValue, Pod, PodPropFlags, Property, PropertyFlags};
use pipewire::spa::sys::*;
use pipewire::spa::utils::{
    Choice, ChoiceEnum, ChoiceFlags, Direction, Fraction, Rectangle, SpaTypes,
};
use pipewire::spa::{self};
use pipewire::stream::{Stream, StreamFlags, StreamListener, StreamRc, StreamState};
use pipewire::sys::{pw_buffer, pw_check_library_version, pw_stream_queue_buffer};
use smithay::backend::allocator::dmabuf::{AsDmabuf, Dmabuf};
use smithay::backend::allocator::format::FormatSet;
use smithay::backend::allocator::gbm::{GbmBuffer, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::Fourcc;
use smithay::backend::drm::DrmNode;
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::utils::{Relocate, RelocateRenderElement};
use smithay::backend::renderer::element::{Element, RenderElement};
use smithay::backend::renderer::sync::SyncPoint;
use smithay::output::{Output, OutputModeSource};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{Interest, LoopHandle, Mode, PostAction};
use smithay::reexports::gbm::Modifier;
use smithay::reexports::rustix::fd::OwnedFd;
use smithay::reexports::rustix::fs::{
    fcntl_add_seals, ftruncate, memfd_create, MemfdFlags, SealFlags,
};
use smithay::utils::{DeviceFd, Logical, Physical, Point, Scale, Size, Transform};
use zbus::object_server::SignalEmitter;

use crate::dbus::mutter_screen_cast::{self, CursorMode};
use crate::niri::{CastTarget, State};
use crate::render_helpers::renderer::NiriCaptureRenderer;
use crate::render_helpers::{clear_dmabuf, encompassing_geo, render_and_download};
use crate::screencasting::CastRenderElement;
use crate::utils::{get_monotonic_time, CastSessionId, CastStreamId};

mod shm_mapping;
use shm_mapping::ShmMapping;
mod render;
use render::RenderCache;

// Give a 0.1 ms allowance for presentation time errors.
const CAST_DELAY_ALLOWANCE: Duration = Duration::from_micros(100);
const DEVICE_CHANGE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_FENCE_POLL_DELAY: Duration = Duration::from_millis(50);
const DMA_RETRY_DELAY: Duration = Duration::from_secs(5);
const MAX_DMA_RETRIES: u32 = 3;
const SHM_BLOCKS: usize = 1;
const SHM_BYTES_PER_PIXEL: usize = 4;

const CURSOR_FORMAT: spa_video_format = SPA_VIDEO_FORMAT_BGRA;
const CURSOR_BPP: u32 = 4;
const CURSOR_WIDTH: u32 = 384;
const CURSOR_HEIGHT: u32 = 384;
const CURSOR_BITMAP_SIZE: usize = (CURSOR_WIDTH * CURSOR_HEIGHT * CURSOR_BPP) as usize;
const CURSOR_META_SIZE: usize =
    mem::size_of::<spa_meta_cursor>() + mem::size_of::<spa_meta_bitmap>() + CURSOR_BITMAP_SIZE;
const BITMAP_META_OFFSET: usize = mem::size_of::<spa_meta_cursor>();
const BITMAP_DATA_OFFSET: usize = mem::size_of::<spa_meta_bitmap>();

#[derive(Clone)]
pub struct CastGbm {
    pub device: GbmDevice<DeviceFd>,
    pub formats: FormatSet,
    pub render_on_primary: bool,
}

#[derive(Clone)]
pub struct CastDevice {
    /// The GPU of the target, including when allocation falls back to SHM or the primary GPU.
    pub node: Option<DrmNode>,
    pub gbm: Option<CastGbm>,
}

#[derive(Default)]
struct CastAllocator {
    gbm: Option<GbmDevice<DeviceFd>>,
    formats: FormatSet,
}

impl From<Option<CastGbm>> for CastAllocator {
    fn from(gbm: Option<CastGbm>) -> Self {
        gbm.map(|gbm| Self {
            gbm: Some(gbm.device),
            formats: gbm.formats,
        })
        .unwrap_or_default()
    }
}

enum DeviceChange {
    /// Do not retire buffers while their old GPU is still writing them.
    WaitingForFrames(CastDevice),
    /// Force a different memory type even if both GPUs offer identical formats/modifiers.
    WaitingForShm(CastDevice),
}

#[derive(Debug, PartialEq, Eq)]
enum DeviceChangeStep {
    Wait,
    NegotiateShm,
    Install,
}

impl DeviceChange {
    fn next_step(
        &self,
        pending_frames: usize,
        allocated_dmabufs: usize,
        has_dma_allocator: bool,
        state: &CastState,
    ) -> DeviceChangeStep {
        match self {
            Self::WaitingForFrames(_) if pending_frames != 0 => DeviceChangeStep::Wait,
            Self::WaitingForFrames(_)
                if allocated_dmabufs != 0
                    || (has_dma_allocator
                        && !matches!(
                            state,
                            CastState::Ready {
                                dma_negotiation: None,
                                ..
                            }
                        )) =>
            {
                DeviceChangeStep::NegotiateShm
            }
            Self::WaitingForFrames(_) => DeviceChangeStep::Install,
            Self::WaitingForShm(_) => {
                if allocated_dmabufs == 0
                    && matches!(
                        state,
                        CastState::Ready {
                            dma_negotiation: None,
                            ..
                        }
                    )
                {
                    DeviceChangeStep::Install
                } else {
                    DeviceChangeStep::Wait
                }
            }
        }
    }

    fn into_device(self) -> CastDevice {
        match self {
            Self::WaitingForFrames(device) | Self::WaitingForShm(device) => device,
        }
    }
}

pub struct PipeWire {
    _context: ContextRc,
    pub core: CoreRc,
    pub token: RegistrationToken,
    event_loop: LoopHandle<'static, State>,
    to_niri: calloop::channel::Sender<PwToNiri>,
}

pub enum PwToNiri {
    StopCast {
        session_id: CastSessionId,
    },
    Redraw {
        stream_id: CastStreamId,
    },
    FallbackToShm {
        stream_id: CastStreamId,
        retry_dma: bool,
    },
    FatalError,
}

pub struct Cast {
    event_loop: LoopHandle<'static, State>,
    pub session_id: CastSessionId,
    pub stream_id: CastStreamId,
    // Listener is dropped before Stream to prevent a use-after-free.
    _listener: StreamListener<()>,
    pub stream: StreamRc,
    pub target: CastTarget,
    pub dynamic_target: bool,
    pub render_on_primary: bool,
    allocator: Rc<RefCell<CastAllocator>>,
    pub device_node: Option<DrmNode>,
    device_change: Option<DeviceChange>,
    device_change_watchdog: Option<RegistrationToken>,
    dma_retry: Option<RegistrationToken>,
    dma_retry_count: u32,
    changing_device: Rc<Cell<bool>>,
    offer_alpha: bool,
    cursor_mode: CursorMode,
    last_frame_time: Duration,
    last_frame_interval: Duration,
    scheduled_redraw: Option<RegistrationToken>,
    cursor_retry: Option<RegistrationToken>,
    // Incremented once per successful frame, stored in buffer meta.
    sequence_counter: u64,
    inner: Rc<RefCell<CastInner>>,
    waiting_for_buffer: Rc<Cell<bool>>,
    to_niri: calloop::channel::Sender<PwToNiri>,
}

/// Mutable `Cast` state shared with PipeWire callbacks.
#[derive(Debug)]
struct CastInner {
    is_active: bool,
    node_id: Option<u32>,
    state: CastState,
    refresh: u32,
    min_time_between_frames: Duration,
    dmabufs: HashMap<i64, Dmabuf>,
    shmbufs: HashMap<i64, Shmbuf>,
    render_cache: RenderCache,
    /// Buffers dequeued from PipeWire in process of rendering.
    ///
    /// This is an ordered list of buffers that we started rendering to and waiting for the
    /// rendering to complete. The completion can be checked from the `SyncPoint`s. The buffers are
    /// stored in order from oldest to newest, and the same ordering should be preserved when
    /// submitting completed buffers to PipeWire.
    rendering_buffers: Vec<(NonNull<pw_buffer>, SyncPoint)>,
}

#[derive(Debug, Clone, Copy)]
struct DmaNegotiation {
    modifier: Modifier,
    plane_count: i32,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum CastState {
    // dma_negotiation = Some(_) means DMA sharing
    // dma_negotiation = None    means SHM sharing
    ResizePending {
        pending_size: Size<u32, Physical>,
    },
    ConfirmationPending {
        size: Size<u32, Physical>,
        alpha: bool,
        dma_negotiation: Option<DmaNegotiation>,
    },
    Ready {
        size: Size<u32, Physical>,
        alpha: bool,
        dma_negotiation: Option<DmaNegotiation>,
        // Lazily-initialized to keep the initialization to a single place.
        damage_tracker: Option<OutputDamageTracker>,
        cursor_damage_tracker: Option<OutputDamageTracker>,
        pending_frame: PendingFrame,
    },
}

/// Changes observed by the damage trackers but not yet sent to the consumer.
///
/// Damage tracking advances even when all PipeWire buffers are in use. Keep changes until a
/// successful render so a final content or cursor update survives backpressure and render errors.
#[derive(Debug, Default)]
struct PendingFrame {
    damaged: bool,
    cursor_damaged: bool,
    last_cursor_location: Option<Point<i32, Physical>>,
}

impl PendingFrame {
    fn update(
        &mut self,
        damaged: bool,
        cursor_damaged: bool,
        cursor_location: Option<Point<i32, Physical>>,
    ) -> bool {
        self.damaged |= damaged;
        self.cursor_damaged |= cursor_damaged;
        self.damaged || self.cursor_damaged || self.last_cursor_location != cursor_location
    }

    fn submitted(&mut self, cursor_location: Option<Point<i32, Physical>>, cursor_updated: bool) {
        self.damaged = false;
        self.cursor_damaged = !cursor_updated;
        self.last_cursor_location = cursor_location;
    }
}

#[derive(PartialEq, Eq)]
pub enum CastSizeChange {
    Ready,
    Pending,
}

/// Data for drawing a cursor either as metadata or embedded.
///
/// The cursor elements are expected to be at the start of the main elements slice. `elem_count` is
/// the count of the pointer elements. This way, the full slice includes both main and cursor
/// elements for embedded mode, and `&elements[elem_count..]` gives just the main elements for
/// metadata mode.
///
/// We have weird borrowed references here in order to support both metadata and embedded cases.
/// The cursor damage tracker needs a slice of impl Element at (0, 0), so we pass it `relocated`
/// (luckily, &impl Element also impls Element). Then, if we need to embed the cursor, we use the
/// full elements slice which starts with non-relocated pointer elements (that we borrow from).
#[derive(Debug)]
pub struct CursorData<'a, E> {
    /// Count of the pointer elements in the slice (index of the first non-pointer element).
    elem_count: usize,
    /// Cursor elements relocated to (0, 0).
    relocated: Vec<RelocateRenderElement<&'a E>>,
    /// Location of the cursor's hotspot in the video buffer.
    location: Point<i32, Physical>,
    /// Location of the cursor's hotspot on the cursor bitmap.
    hotspot: Point<i32, Physical>,
    /// Size of the elements' encompassing geo.
    size: Size<i32, Physical>,
    /// Scale the elements should be rendered at.
    scale: Scale<f64>,
}

impl<'a, E: Element> CursorData<'a, E> {
    pub fn compute(
        elements: &'a [E],
        elem_count: usize,
        location: Point<f64, Logical>,
        scale: Scale<f64>,
    ) -> Self {
        let pointer_elements = &elements[..elem_count];
        let location = location.to_physical_precise_round(scale);

        let geo = encompassing_geo(scale, pointer_elements.iter());
        let relocated = Vec::from_iter(pointer_elements.iter().map(|elem| {
            RelocateRenderElement::from_element(elem, geo.loc.upscale(-1), Relocate::Relative)
        }));

        Self {
            elem_count,
            relocated,
            location,
            hotspot: location - geo.loc,
            size: geo.size,
            scale,
        }
    }
}

fn make_video_params(
    format: VideoFormat,
    modifiers: &[Modifier],
    size: Size<u32, Physical>,
    refresh: u32,
) -> pod::Object {
    let mut properties = vec![
        pod::property!(FormatProperties::MediaType, Id, MediaType::Video),
        pod::property!(FormatProperties::MediaSubtype, Id, MediaSubtype::Raw),
        pod::property!(
            FormatProperties::VideoSize,
            Rectangle,
            Rectangle {
                width: size.w,
                height: size.h,
            }
        ),
        pod::property!(
            FormatProperties::VideoFramerate,
            Fraction,
            Fraction { num: 0, denom: 1 }
        ),
        pod::property!(
            FormatProperties::VideoMaxFramerate,
            Choice,
            Range,
            Fraction,
            Fraction {
                num: refresh,
                denom: 1000
            },
            Fraction { num: 1, denom: 1 },
            Fraction {
                num: refresh,
                denom: 1000
            }
        ),
        pod::property!(FormatProperties::VideoFormat, Id, format),
    ];

    if !modifiers.is_empty() {
        let dont_fixate = if modifiers.len() > 1 {
            PropertyFlags::DONT_FIXATE
        } else {
            PropertyFlags::empty()
        };
        let flags = PropertyFlags::MANDATORY | dont_fixate;
        let modifiers_i64 = modifiers
            .iter()
            .map(|m| u64::from(*m) as i64)
            .collect::<Vec<_>>();

        let prop = Property {
            key: FormatProperties::VideoModifier.as_raw(),
            flags,
            value: pod::Value::Choice(ChoiceValue::Long(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Enum {
                    default: modifiers_i64[0],
                    alternatives: modifiers_i64,
                },
            ))),
        };

        properties.push(prop);
    }

    pod::Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties,
    }
}

fn make_initial_video_params(
    possible_modifiers: &FormatSet,
    size: Size<u32, Physical>,
    refresh: u32,
    alpha: bool,
) -> Vec<pod::Object> {
    let mut rv = Vec::new();

    let mut push_alpha = |alpha| {
        let format = if alpha {
            VideoFormat::BGRA
        } else {
            VideoFormat::BGRx
        };

        let fourcc = if alpha {
            Fourcc::Argb8888
        } else {
            Fourcc::Xrgb8888
        };

        let modifiers: Vec<_> = possible_modifiers
            .iter()
            .filter_map(|f| (f.code == fourcc).then_some(f.modifier))
            .collect();

        trace!("offering: {modifiers:?}");

        if !modifiers.is_empty() {
            rv.push(make_video_params(format, &modifiers, size, refresh));
        }
        rv.push(make_video_params(format, &[], size, refresh));
    };

    if alpha {
        push_alpha(true);
    }
    push_alpha(false);

    rv
}

macro_rules! make_params {
    ($params:ident, $formats:expr, $size:expr, $refresh:expr, $alpha:expr) => {
        let $params = make_initial_video_params($formats, $size, $refresh, $alpha);
        let mut bufs = [const { Vec::new() }; 4]; // Maximum possible params len.
        let mut $params: Vec<_> = $params
            .into_iter()
            .zip(&mut bufs)
            .map(|(obj, buf)| make_pod(buf, obj))
            .collect();
    };
}

impl PipeWire {
    pub fn new(
        event_loop: LoopHandle<'static, State>,
        to_niri: calloop::channel::Sender<PwToNiri>,
    ) -> anyhow::Result<Self> {
        let main_loop = MainLoopRc::new(None).context("error creating MainLoop")?;
        let context = ContextRc::new(&main_loop, None).context("error creating Context")?;
        let core = context.connect_rc(None).context("error creating Core")?;

        let to_niri_ = to_niri.clone();
        let listener = core
            .add_listener_local()
            .error(move |id, seq, res, message| {
                warn!(id, seq, res, message, "pw error");

                // Reset PipeWire on connection errors.
                if id == PW_ID_CORE && res == -32 {
                    if let Err(err) = to_niri_.send(PwToNiri::FatalError) {
                        warn!("error sending FatalError to niri: {err:?}");
                    }
                }
            })
            .register();
        mem::forget(listener);

        struct AsFdWrapper(MainLoopRc);
        impl AsFd for AsFdWrapper {
            fn as_fd(&self) -> BorrowedFd<'_> {
                self.0.loop_().fd()
            }
        }
        let generic = Generic::new(AsFdWrapper(main_loop), Interest::READ, Mode::Level);
        let token = event_loop
            .insert_source(generic, move |_, wrapper, _| {
                let _span = tracy_client::span!("pipewire iteration");
                wrapper.0.loop_().iterate(Timeout::None);
                Ok(PostAction::Continue)
            })
            .unwrap();

        Ok(Self {
            _context: context,
            core,
            token,
            event_loop,
            to_niri,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_cast(
        &self,
        device: CastDevice,
        session_id: CastSessionId,
        stream_id: CastStreamId,
        target: CastTarget,
        size: Size<i32, Physical>,
        refresh: u32,
        alpha: bool,
        mut cursor_mode: CursorMode,
        signal_ctx: SignalEmitter<'static>,
    ) -> anyhow::Result<Cast> {
        let _span = tracy_client::span!("PipeWire::start_cast");
        let _span = debug_span!("start_cast", %session_id).entered();

        let to_niri_ = self.to_niri.clone();
        let stop_cast = move || {
            if let Err(err) = to_niri_.send(PwToNiri::StopCast { session_id }) {
                warn!("error sending StopCast to niri: {err:?}");
            }
        };
        let to_niri_ = self.to_niri.clone();
        let redraw = move || {
            if let Err(err) = to_niri_.send(PwToNiri::Redraw { stream_id }) {
                warn!("error sending Redraw to niri: {err:?}");
            }
        };
        let to_niri_ = self.to_niri.clone();
        let fallback_to_shm = move |retry_dma| {
            if let Err(err) = to_niri_.send(PwToNiri::FallbackToShm {
                stream_id,
                retry_dma,
            }) {
                warn!("error sending FallbackToShm to niri: {err:?}");
            }
        };
        let redraw_ = redraw.clone();
        let redraw_process = redraw.clone();
        let redraw_remove = redraw.clone();
        let waiting_for_buffer = Rc::new(Cell::new(false));
        let changing_device = Rc::new(Cell::new(false));

        let stream = StreamRc::new(
            self.core.clone(),
            "niri-screen-cast-src",
            PropertiesBox::new(),
        )
        .context("error creating Stream")?;

        if cursor_mode == CursorMode::Metadata && !pw_version_supports_cursor_metadata() {
            debug!(
                "metadata cursor mode requested, but PipeWire is too old (need >= 1.4.8); \
                 switching to embedded cursor"
            );
            cursor_mode = CursorMode::Embedded;
        }

        let pending_size = Size::from((size.w as u32, size.h as u32));

        let render_on_primary = device.gbm.as_ref().is_some_and(|gbm| gbm.render_on_primary);
        let allocator = Rc::new(RefCell::new(CastAllocator::from(device.gbm)));

        // Like in good old wayland-rs times...
        let inner = Rc::new(RefCell::new(CastInner {
            is_active: false,
            node_id: None,
            state: CastState::ResizePending { pending_size },
            refresh,
            min_time_between_frames: Duration::ZERO,
            dmabufs: HashMap::new(),
            shmbufs: HashMap::new(),
            render_cache: RenderCache::default(),
            rendering_buffers: Vec::new(),
        }));

        let listener = stream
            .add_local_listener_with_user_data(())
            .state_changed({
                let inner = inner.clone();
                let stop_cast = stop_cast.clone();
                move |stream, (), old, new| {
                    let _span = debug_span!("state_changed", %stream_id).entered();
                    debug!("{old:?} -> {new:?}");
                    let mut inner = inner.borrow_mut();

                    match new {
                        StreamState::Paused => {
                            if inner.node_id.is_none() {
                                let id = stream.node_id();
                                inner.node_id = Some(id);
                                debug!("sending signal with {id}");

                                let _span = tracy_client::span!("sending PipeWireStreamAdded");
                                async_io::block_on(async {
                                    let res = mutter_screen_cast::Stream::pipe_wire_stream_added(
                                        &signal_ctx,
                                        id,
                                    )
                                    .await;

                                    if let Err(err) = res {
                                        warn!("error sending PipeWireStreamAdded: {err:?}");
                                        stop_cast();
                                    }
                                });
                            }

                            inner.is_active = false;
                        }
                        StreamState::Error(_) => {
                            if inner.is_active {
                                inner.is_active = false;
                                stop_cast();
                            }
                        }
                        StreamState::Unconnected => (),
                        StreamState::Connecting => (),
                        StreamState::Streaming => {
                            inner.is_active = true;
                            redraw();
                        }
                    }
                }
            })
            .process({
                let waiting_for_buffer = waiting_for_buffer.clone();
                move |_stream, ()| {
                    // Retry the final update once the consumer returns a buffer. Do not
                    // continuously redraw idle streams on every PipeWire process event.
                    if waiting_for_buffer.replace(false) {
                        redraw_process();
                    }
                }
            })
            .param_changed({
                let inner = inner.clone();
                let stop_cast = stop_cast.clone();
                let allocator = allocator.clone();
                let fallback_to_shm = fallback_to_shm.clone();
                move |stream, (), id, pod| {
                    let id = ParamType::from_raw(id);
                    trace!(%stream_id, ?id, "param_changed");
                    let mut inner = inner.borrow_mut();
                    let inner = &mut *inner;

                    if id != ParamType::Format {
                        return;
                    }

                    let _span = debug_span!("param_changed", %stream_id).entered();

                    let Some(pod) = pod else { return };

                    let (m_type, m_subtype) = match parse_format(pod) {
                        Ok(x) => x,
                        Err(err) => {
                            warn!("error parsing format: {err:?}");
                            return;
                        }
                    };

                    if m_type != MediaType::Video || m_subtype != MediaSubtype::Raw {
                        return;
                    }

                    let mut format = VideoInfoRaw::new();
                    format.parse(pod).unwrap();
                    debug!("got format = {format:?}");

                    let format_size = Size::from((format.size().width, format.size().height));

                    let state = &mut inner.state;
                    if format_size != state.expected_format_size() {
                        if !matches!(&*state, CastState::ResizePending { .. }) {
                            warn!("wrong size, but we're not resizing");
                            stop_cast();
                            return;
                        }

                        debug!("wrong size, waiting");
                        return;
                    }

                    let format_has_alpha = format.format() == VideoFormat::BGRA;
                    let fourcc = if format_has_alpha {
                        Fourcc::Argb8888
                    } else {
                        Fourcc::Xrgb8888
                    };

                    let max_frame_rate = format.max_framerate();
                    let min_frame_time = Duration::from_micros(
                        1_000_000 * u64::from(max_frame_rate.denom) / u64::from(max_frame_rate.num),
                    );
                    inner.min_time_between_frames = min_frame_time;

                    // We have following cases when param_changed:
                    //
                    // 1. Modifier exists and its flags contain DONT_FIXATE
                    //
                    //    Do test allocation, set CastState to ConfirmationPending and send
                    //    param again.
                    //
                    // 2. Modifier exists and it doesn't need fixation
                    //
                    //    Do test allocation to ensure the modifier work, then set CastState to
                    //    Ready. Then set buffer to DMA.
                    //
                    // 3. Modifier doesn't exist
                    //
                    //    Set CastState to Ready and set buffer to SHM.

                    let object = pod.as_object().unwrap();
                    let prop_modifier =
                        object.find_prop(spa::utils::Id(FormatProperties::VideoModifier.0));

                    let allocator = allocator.borrow();
                    let gbm = &allocator.gbm;
                    let formats = &allocator.formats;
                    // A late reply to the previous DMA offer must not revive the old allocator
                    // while a GPU migration is negotiating SHM.
                    if prop_modifier.is_some() && gbm.is_none() {
                        trace!("ignoring stale DMA format during SHM negotiation");
                        return;
                    }

                    match prop_modifier {
                        Some(prop_modifier)
                            if prop_modifier.flags().contains(PodPropFlags::DONT_FIXATE) =>
                        {
                            debug!(flags = ?prop_modifier.flags(), "fixating the modifier");

                            let Some(gbm) = &gbm else {
                                error!("negotiated dmabuf without gbm");
                                stop_cast();
                                return;
                            };

                            let pod_modifier = prop_modifier.value();
                            let modifiers = match parse_modifier_candidates(pod_modifier) {
                                Ok(modifiers) => modifiers,
                                Err(err) => {
                                    warn!("invalid modifier property: {err:?}");
                                    stop_cast();
                                    return;
                                }
                            };

                            let (modifier, plane_count) = match find_preferred_modifier(
                                gbm,
                                format_size,
                                fourcc,
                                modifiers,
                            ) {
                                Ok(x) => x,
                                Err(err) => {
                                    warn!("couldn't find preferred modifier, trying SHM: {err:?}");
                                    fallback_to_shm(false);
                                    return;
                                }
                            };

                            debug!(
                                "allocation successful \
                                     (modifier={modifier:?}, plane_count={plane_count}), \
                                     moving to confirmation pending"
                            );

                            *state = CastState::ConfirmationPending {
                                size: format_size,
                                alpha: format_has_alpha,
                                dma_negotiation: Some(DmaNegotiation {
                                    modifier,
                                    plane_count: plane_count as i32,
                                }),
                            };

                            let o = make_video_params(
                                format.format(),
                                &[modifier],
                                format_size,
                                inner.refresh,
                            );
                            let mut b = Vec::new();
                            let pod = make_pod(&mut b, o);

                            make_params!(
                                params,
                                &formats,
                                format_size,
                                inner.refresh,
                                format_has_alpha
                            );
                            params.insert(0, pod);

                            if let Err(err) = stream.update_params(&mut params) {
                                warn!("error updating stream params: {err:?}");
                                stop_cast();
                            }

                            return;
                        }
                        _ => (),
                    }

                    let o1 = if prop_modifier.is_some() {
                        // Verify that alpha and modifier didn't change.
                        let plane_count = match &*state {
                            CastState::ConfirmationPending {
                                size,
                                alpha,
                                dma_negotiation: Some(dma_negotiation),
                            }
                            | CastState::Ready {
                                size,
                                alpha,
                                dma_negotiation: Some(dma_negotiation),
                                ..
                            } if *alpha == format_has_alpha
                                && dma_negotiation.modifier
                                    == Modifier::from(format.modifier()) =>
                            {
                                let size = *size;
                                let alpha = *alpha;
                                let dma_negotiation = *dma_negotiation;

                                let (damage_tracker, cursor_damage_tracker) =
                                    if let CastState::Ready {
                                        damage_tracker,
                                        cursor_damage_tracker,
                                        ..
                                    } = &mut *state
                                    {
                                        (damage_tracker.take(), cursor_damage_tracker.take())
                                    } else {
                                        (None, None)
                                    };

                                debug!("moving to ready state");

                                *state = CastState::Ready {
                                    size,
                                    alpha,
                                    dma_negotiation: Some(dma_negotiation),
                                    damage_tracker,
                                    cursor_damage_tracker,
                                    pending_frame: PendingFrame::default(),
                                };

                                dma_negotiation.plane_count
                            }
                            _ => {
                                let Some(gbm) = &gbm else {
                                    error!("negotiated dmabuf without gbm");
                                    stop_cast();
                                    return;
                                };

                                // We're negotiating a single modifier, or alpha or modifier
                                // changed, so we need to do a test allocation.
                                let (modifier, plane_count) = match find_preferred_modifier(
                                    gbm,
                                    format_size,
                                    fourcc,
                                    vec![format.modifier() as i64],
                                ) {
                                    Ok(x) => x,
                                    Err(err) => {
                                        warn!("test allocation failed, trying SHM: {err:?}");
                                        fallback_to_shm(false);
                                        return;
                                    }
                                };

                                debug!(
                                    "allocation successful \
                                         (modifier={modifier:?}, plane_count={plane_count}), \
                                         moving to ready"
                                );

                                *state = CastState::Ready {
                                    size: format_size,
                                    alpha: format_has_alpha,
                                    dma_negotiation: Some(DmaNegotiation {
                                        modifier,
                                        plane_count: plane_count as i32,
                                    }),
                                    damage_tracker: None,
                                    cursor_damage_tracker: None,
                                    pending_frame: PendingFrame::default(),
                                };

                                plane_count as i32
                            }
                        };

                        pod::object!(
                            SpaTypes::ObjectParamBuffers,
                            ParamType::Buffers,
                            Property::new(
                                SPA_PARAM_BUFFERS_buffers,
                                pod::Value::Choice(ChoiceValue::Int(Choice(
                                    ChoiceFlags::empty(),
                                    ChoiceEnum::Range {
                                        default: 8,
                                        min: 2,
                                        max: 16
                                    }
                                ))),
                            ),
                            Property::new(SPA_PARAM_BUFFERS_blocks, pod::Value::Int(plane_count)),
                            Property::new(
                                SPA_PARAM_BUFFERS_dataType,
                                pod::Value::Choice(ChoiceValue::Int(Choice(
                                    ChoiceFlags::empty(),
                                    ChoiceEnum::Flags {
                                        default: 1 << DataType::DmaBuf.as_raw(),
                                        flags: vec![1 << DataType::DmaBuf.as_raw()],
                                    },
                                ))),
                            ),
                        )
                    } else {
                        debug!("negotiated inefficient shm stream, moving to ready state");

                        *state = CastState::Ready {
                            size: format_size,
                            alpha: format_has_alpha,
                            dma_negotiation: None,
                            damage_tracker: None,
                            cursor_damage_tracker: None,
                            pending_frame: PendingFrame::default(),
                        };
                        pod::object!(
                            SpaTypes::ObjectParamBuffers,
                            ParamType::Buffers,
                            Property::new(
                                SPA_PARAM_BUFFERS_buffers,
                                pod::Value::Choice(ChoiceValue::Int(Choice(
                                    ChoiceFlags::empty(),
                                    ChoiceEnum::Range {
                                        default: 8,
                                        min: 2,
                                        max: 16
                                    }
                                ))),
                            ),
                            Property::new(
                                SPA_PARAM_BUFFERS_blocks,
                                pod::Value::Int(SHM_BLOCKS as i32),
                            ),
                            Property::new(
                                SPA_PARAM_BUFFERS_dataType,
                                pod::Value::Choice(ChoiceValue::Int(Choice(
                                    ChoiceFlags::empty(),
                                    ChoiceEnum::Flags {
                                        default: 1 << DataType::MemFd.as_raw(),
                                        flags: vec![1 << DataType::MemFd.as_raw()],
                                    },
                                ))),
                            ),
                        )
                    };

                    let o2 = pod::object!(
                        SpaTypes::ObjectParamMeta,
                        ParamType::Meta,
                        Property::new(
                            SPA_PARAM_META_type,
                            pod::Value::Id(spa::utils::Id(SPA_META_Header))
                        ),
                        Property::new(
                            SPA_PARAM_META_size,
                            pod::Value::Int(size_of::<spa_meta_header>() as i32)
                        ),
                    );

                    let mut b1 = vec![];
                    let mut b2 = vec![];

                    let mut params = vec![make_pod(&mut b1, o1), make_pod(&mut b2, o2)];

                    let mut b_cursor = vec![];
                    if cursor_mode == CursorMode::Metadata {
                        let o_cursor = pod::object!(
                            SpaTypes::ObjectParamMeta,
                            ParamType::Meta,
                            Property::new(
                                SPA_PARAM_META_type,
                                pod::Value::Id(spa::utils::Id(SPA_META_Cursor))
                            ),
                            Property::new(
                                SPA_PARAM_META_size,
                                pod::Value::Int(CURSOR_META_SIZE as i32)
                            ),
                        );
                        params.push(make_pod(&mut b_cursor, o_cursor));
                    }

                    if let Err(err) = stream.update_params(&mut params) {
                        warn!("error updating stream params: {err:?}");
                        stop_cast();
                    }
                }
            })
            .add_buffer({
                let inner = inner.clone();
                let allocator = allocator.clone();
                let stop_cast = stop_cast.clone();
                let fallback_to_shm = fallback_to_shm.clone();
                move |stream, (), buffer| {
                    let _span = debug_span!("add_buffer", %stream_id).entered();

                    let allocator = allocator.borrow();
                    match unsafe {
                        inner
                            .borrow_mut()
                            .on_add_buffer(allocator.gbm.as_ref(), buffer)
                    } {
                        Ok(redraw) => {
                            // During size re-negotiation, the stream sometimes just keeps
                            // running, in which case we may need to force a redraw once we got
                            // a newly sized buffer.
                            if redraw && stream.state() == StreamState::Streaming {
                                redraw_();
                            }
                        }
                        Err(err) => {
                            warn!("error adding pw buffer: {err:?}");
                            if allocator.gbm.is_some() {
                                fallback_to_shm(true);
                            } else {
                                stop_cast();
                            }
                        }
                    };
                }
            })
            .remove_buffer({
                let inner = inner.clone();
                let changing_device = changing_device.clone();
                move |_stream, (), buffer| {
                    let _span = debug_span!("remove_buffer", %stream_id).entered();

                    let last_dma_removed = unsafe {
                        let mut inner = inner.borrow_mut();
                        let had_dma = !inner.dmabufs.is_empty();
                        inner.on_remove_buffer(buffer);
                        had_dma && inner.dmabufs.is_empty()
                    };
                    if last_dma_removed && changing_device.get() {
                        redraw_remove();
                    }
                }
            })
            .register()
            .unwrap();

        trace!("starting pw stream with size={pending_size:?}, refresh={refresh:?}");

        make_params!(
            params,
            &allocator.borrow().formats,
            pending_size,
            refresh,
            alpha
        );
        stream
            .connect(
                Direction::Output,
                None,
                StreamFlags::DRIVER | StreamFlags::ALLOC_BUFFERS,
                &mut params,
            )
            .context("error connecting stream")?;

        let cast = Cast {
            event_loop: self.event_loop.clone(),
            session_id,
            stream_id,
            stream,
            _listener: listener,
            target,
            dynamic_target: false,
            render_on_primary,
            allocator,
            device_node: device.node,
            device_change: None,
            device_change_watchdog: None,
            dma_retry: None,
            dma_retry_count: 0,
            changing_device,
            offer_alpha: alpha,
            cursor_mode,
            last_frame_time: Duration::ZERO,
            last_frame_interval: Duration::ZERO,
            scheduled_redraw: None,
            cursor_retry: None,
            sequence_counter: 0,
            inner,
            waiting_for_buffer,
            to_niri: self.to_niri.clone(),
        };
        Ok(cast)
    }
}

impl Cast {
    pub fn is_active(&self) -> bool {
        self.device_change.is_none() && self.inner.borrow().is_active
    }

    pub fn size(&self) -> Size<i32, Physical> {
        let size = self.inner.borrow().state.expected_format_size();
        Size::from((size.w as i32, size.h as i32))
    }

    pub fn set_device(&mut self, device: CastDevice) -> anyhow::Result<()> {
        self.remove_device_change_watchdog();
        if let Some(token) = self.dma_retry.take() {
            self.event_loop.remove(token);
        }
        if self.device_node != device.node {
            self.dma_retry_count = 0;
        }
        self.device_node = device.node;
        self.device_change = Some(DeviceChange::WaitingForFrames(device));
        self.changing_device.set(true);
        self.progress_device_change()
    }

    pub fn fallback_to_shm(&mut self, retry_dma: bool) -> anyhow::Result<()> {
        // Several add_buffer callbacks can report the same allocation failure.
        if matches!(
            &self.device_change,
            Some(DeviceChange::WaitingForFrames(CastDevice { gbm: None, .. }))
                | Some(DeviceChange::WaitingForShm(CastDevice { gbm: None, .. }))
        ) || (self.allocator.borrow().gbm.is_none() && self.device_change.is_none())
        {
            return Ok(());
        }
        let retry_device = if retry_dma && self.dma_retry_count < MAX_DMA_RETRIES {
            let allocator = self.allocator.borrow();
            allocator.gbm.as_ref().map(|device| CastDevice {
                node: self.device_node,
                gbm: Some(CastGbm {
                    device: device.clone(),
                    formats: allocator.formats.clone(),
                    render_on_primary: self.render_on_primary,
                }),
            })
        } else {
            None
        };
        self.set_device(CastDevice {
            node: self.device_node,
            gbm: None,
        })?;
        if let Some(device) = retry_device {
            self.schedule_dma_retry(device);
        }
        Ok(())
    }

    fn schedule_dma_retry(&mut self, device: CastDevice) {
        let stream_id = self.stream_id;
        let delay = DMA_RETRY_DELAY * (1 << self.dma_retry_count);
        self.dma_retry_count += 1;
        let token = self
            .event_loop
            .insert_source(Timer::from_duration(delay), move |_, _, state| {
                let Some(cast) = state
                    .niri
                    .casting
                    .casts
                    .iter_mut()
                    .find(|cast| cast.stream_id == stream_id)
                else {
                    return TimeoutAction::Drop;
                };
                // Do not interrupt retirement of the old buffers, including unfinished writes.
                if cast.device_change.is_some() {
                    return TimeoutAction::ToDuration(delay);
                }
                cast.dma_retry = None;
                let session_id = cast.session_id;
                debug!(%stream_id, "retrying screencast DMA allocation after runtime failure");
                if let Err(err) = cast.set_device(device.clone()) {
                    warn!(%stream_id, "error retrying screencast GPU: {err:?}");
                    state.niri.stop_cast(session_id);
                } else {
                    state.redraw_cast(stream_id);
                }
                TimeoutAction::Drop
            })
            .unwrap();
        self.dma_retry = Some(token);
    }

    /// Retire the old GPU's buffers through SHM before advertising the new allocator.
    pub fn progress_device_change(&mut self) -> anyhow::Result<()> {
        let Some(change) = self.device_change.take() else {
            return Ok(());
        };
        self.queue_completed_buffers();
        let inner = self.inner.borrow();
        let step = change.next_step(
            inner.rendering_buffers.len(),
            inner.dmabufs.len(),
            self.allocator.borrow().gbm.is_some(),
            &inner.state,
        );
        drop(inner);
        match step {
            DeviceChangeStep::Wait => {
                self.device_change = Some(change);
                return Ok(());
            }
            DeviceChangeStep::NegotiateShm => {
                *self.allocator.borrow_mut() = CastAllocator::default();
                self.device_change = Some(DeviceChange::WaitingForShm(change.into_device()));
                self.start_device_change_watchdog();
                return self.renegotiate_device(true);
            }
            DeviceChangeStep::Install => (),
        }
        self.remove_device_change_watchdog();
        let device = change.into_device();
        self.changing_device.set(false);
        self.render_on_primary = device.gbm.as_ref().is_some_and(|gbm| gbm.render_on_primary);
        *self.allocator.borrow_mut() = CastAllocator::from(device.gbm);
        self.renegotiate_device(false)
    }

    fn start_device_change_watchdog(&mut self) {
        let stream_id = self.stream_id;
        let token = self
            .event_loop
            .insert_source(Timer::from_duration(DEVICE_CHANGE_TIMEOUT), move |_, _, state| {
                let Some(cast) = state
                    .niri
                    .casting
                    .casts
                    .iter_mut()
                    .find(|cast| cast.stream_id == stream_id)
                else {
                    return TimeoutAction::Drop;
                };
                cast.device_change_watchdog = None;
                if matches!(cast.device_change, Some(DeviceChange::WaitingForShm(_))) {
                    // GPU writes were drained before SHM negotiation. It is safe to
                    // disconnect here, but not to force-install over the old DMA buffers.
                    warn!(%stream_id, "screencast GPU renegotiation timed out, stopping session");
                    let session_id = cast.session_id;
                    state.niri.stop_cast(session_id);
                }
                TimeoutAction::Drop
            })
            .unwrap();
        self.device_change_watchdog = Some(token);
    }

    fn remove_device_change_watchdog(&mut self) {
        if let Some(token) = self.device_change_watchdog.take() {
            self.event_loop.remove(token);
        }
    }

    fn renegotiate_device(&mut self, retire_dma: bool) -> anyhow::Result<()> {
        let mut inner = self.inner.borrow_mut();
        let size = inner.state.expected_format_size();
        if retire_dma {
            inner.state = CastState::ResizePending { pending_size: size };
        } else {
            // The consumer may keep the same SHM format after receiving the new DMA offer.
            // Keep it usable even if PipeWire does not emit another format-changed event.
            inner.reset_damage();
        }
        let refresh = inner.refresh;
        drop(inner);
        self.waiting_for_buffer.set(false);
        make_params!(
            params,
            &self.allocator.borrow().formats,
            size,
            refresh,
            self.offer_alpha
        );
        self.stream
            .update_params(&mut params)
            .context("error renegotiating screencast GPU")
    }

    pub fn node_id(&self) -> Option<u32> {
        self.inner.borrow().node_id
    }

    pub fn ensure_size(&self, size: Size<i32, Physical>) -> anyhow::Result<CastSizeChange> {
        if self.device_change.is_some() {
            return Ok(CastSizeChange::Pending);
        }
        let mut inner = self.inner.borrow_mut();

        let new_size = Size::from((size.w as u32, size.h as u32));

        let state = &mut inner.state;
        if matches!(state, CastState::Ready { size, .. } if *size == new_size) {
            return Ok(CastSizeChange::Ready);
        }

        if state.pending_size() == Some(new_size) {
            debug!("stream size still hasn't changed, skipping frame");
            return Ok(CastSizeChange::Pending);
        }

        let _span = tracy_client::span!("Cast::ensure_size");
        debug!("cast size changed, updating stream size");

        *state = CastState::ResizePending {
            pending_size: new_size,
        };

        make_params!(
            params,
            &self.allocator.borrow().formats,
            new_size,
            inner.refresh,
            self.offer_alpha
        );
        self.stream
            .update_params(&mut params)
            .context("error updating stream params")?;

        Ok(CastSizeChange::Pending)
    }

    pub fn set_refresh(&mut self, refresh: u32) -> anyhow::Result<()> {
        let mut inner = self.inner.borrow_mut();

        if inner.refresh == refresh {
            return Ok(());
        }

        let _span = tracy_client::span!("Cast::set_refresh");
        debug!("cast FPS changed, updating stream FPS");
        inner.refresh = refresh;
        if self.device_change.is_some() {
            return Ok(());
        }

        let size = inner.state.expected_format_size();
        make_params!(
            params,
            &self.allocator.borrow().formats,
            size,
            refresh,
            self.offer_alpha
        );
        self.stream
            .update_params(&mut params)
            .context("error updating stream params")?;

        Ok(())
    }

    pub fn record_frame_time(&mut self, recorded: Duration) {
        let interval = self.inner.borrow().min_time_between_frames;
        let ideal = self.last_frame_time + interval;

        // Absorb small (< 1 frame) differences in output refresh interval vs. screencast framerate
        // to keep the screencast time base consistent instead of shifting forward every frame.
        //
        // After a missed interval or a rate change, restart instead of catching up in a burst.
        self.last_frame_time = if self.last_frame_interval == interval
            && recorded >= self.last_frame_time
            && recorded < ideal + interval
        {
            ideal
        } else {
            recorded
        };
        self.last_frame_interval = interval;
    }

    fn compute_extra_delay(&self, target_frame_time: Duration) -> Duration {
        let inner = self.inner.borrow();

        let last = self.last_frame_time;
        let min = inner.min_time_between_frames;

        if last.is_zero() {
            trace!(?target_frame_time, ?last, "last is zero, recording");
            return Duration::ZERO;
        }

        if target_frame_time < last {
            // Record frame with a warning; in case it was an overflow this will fix it.
            warn!(
                ?target_frame_time,
                ?last,
                "target frame time is below last, did it overflow or did we mispredict?"
            );
            return Duration::ZERO;
        }

        let diff = target_frame_time - last;
        if diff < min {
            let delay = min - diff;
            trace!(
                ?target_frame_time,
                ?last,
                "frame is too soon: min={min:?}, delay={:?}",
                delay
            );
            return delay;
        } else {
            trace!("overshoot={:?}", diff - min);
        }

        Duration::ZERO
    }

    fn schedule_redraw(&mut self, output: Output, target_time: Duration) {
        if self.scheduled_redraw.is_some() {
            return;
        }

        let now = get_monotonic_time();
        let duration = target_time.saturating_sub(now);
        let timer = Timer::from_duration(duration);
        let token = self
            .event_loop
            .insert_source(timer, move |_, _, state| {
                // Guard against output disconnecting before the timer has a chance to run.
                if state.niri.output_state.contains_key(&output) {
                    state.niri.queue_redraw(&output);
                }

                TimeoutAction::Drop
            })
            .unwrap();
        self.scheduled_redraw = Some(token);
    }

    fn remove_scheduled_redraw(&mut self) {
        if let Some(token) = self.scheduled_redraw.take() {
            self.event_loop.remove(token);
        }
    }

    fn schedule_cursor_retry(&mut self) {
        if self.cursor_retry.is_some() {
            return;
        }
        let stream_id = self.stream_id;
        let delay = self
            .inner
            .borrow()
            .min_time_between_frames
            .max(Duration::from_millis(16));
        let token = self
            .event_loop
            .insert_source(Timer::from_duration(delay), move |_, _, state| {
                let Some(cast) = state
                    .niri
                    .casting
                    .casts
                    .iter_mut()
                    .find(|cast| cast.stream_id == stream_id)
                else {
                    return TimeoutAction::Drop;
                };
                cast.cursor_retry = None;
                state.redraw_cast(stream_id);
                TimeoutAction::Drop
            })
            .unwrap();
        self.cursor_retry = Some(token);
    }

    /// Checks whether this frame should be skipped because it's too soon.
    ///
    /// If the frame should be skipped, schedules a redraw and returns `true`. Otherwise, removes a
    /// scheduled redraw, if any, and returns `false`.
    ///
    /// When this method returns `false`, the calling code is assumed to follow up with
    /// [`Cast::dequeue_buffer_and_render()`].
    pub fn check_time_and_schedule(
        &mut self,
        output: &Output,
        target_frame_time: Duration,
    ) -> bool {
        let delay = self.compute_extra_delay(target_frame_time);
        if delay >= CAST_DELAY_ALLOWANCE {
            trace!("delay >= allowance, scheduling redraw");
            self.schedule_redraw(output.clone(), target_frame_time + delay);
            true
        } else {
            self.remove_scheduled_redraw();
            false
        }
    }

    fn dequeue_available_buffer(&mut self) -> Option<NonNull<pw_buffer>> {
        let buffer = unsafe { NonNull::new(self.stream.dequeue_raw_buffer()) };
        self.waiting_for_buffer.set(buffer.is_none());
        buffer
    }

    fn queue_completed_buffers(&mut self) {
        let mut inner = self.inner.borrow_mut();

        // We want to queue buffers in order, so find the first still-rendering buffer, and queue
        // everything up to that. Even if there are completed buffers past the first
        // still-rendering buffer, we do not want to queue them, since that would send frames out
        // of order.
        let first_in_progress_idx = inner
            .rendering_buffers
            .iter()
            .position(|(_, sync)| !sync.is_reached())
            .unwrap_or(inner.rendering_buffers.len());

        for (buffer, _) in inner.rendering_buffers.drain(..first_in_progress_idx) {
            trace!("queueing completed buffer");
            unsafe {
                pw_stream_queue_buffer(self.stream.as_raw_ptr(), buffer.as_ptr());
            }
        }
        if first_in_progress_idx != 0
            && inner.rendering_buffers.is_empty()
            && self.changing_device.get()
        {
            let _ = self.to_niri.send(PwToNiri::Redraw {
                stream_id: self.stream_id,
            });
        }
    }

    unsafe fn queue_after_sync(&mut self, pw_buffer: NonNull<pw_buffer>, sync_point: SyncPoint) {
        let _span = tracy_client::span!("Cast::queue_after_sync");

        let mut inner = self.inner.borrow_mut();

        let sync_fd = sync_point.export();
        // Export can fail while the GPU is still writing. Preserve that dependency; in
        // particular, a device migration must never mistake it for a completed buffer.
        let needs_poll = sync_fd.is_none() && !sync_point.is_reached();
        let poll_sync = needs_poll.then(|| sync_point.clone());

        inner.rendering_buffers.push((pw_buffer, sync_point));
        drop(inner);

        match sync_fd {
            None => {
                self.queue_completed_buffers();
                if let Some(poll_sync) = poll_sync {
                    let stream_id = self.stream_id;
                    let mut delay = Duration::from_millis(1);
                    let started = Instant::now();
                    let mut warned = false;
                    self.event_loop
                        .insert_source(Timer::from_duration(delay), move |_, _, state| {
                            let Some(cast) = state
                                .niri
                                .casting
                                .casts
                                .iter_mut()
                                .find(|cast| cast.stream_id == stream_id)
                            else {
                                return TimeoutAction::Drop;
                            };
                            // Check this fence before draining: completion between the two
                            // checks must not drop its last wakeup with a buffer still queued.
                            let completed = poll_sync.is_reached();
                            cast.queue_completed_buffers();
                            if !completed {
                                if !warned && started.elapsed() >= DEVICE_CHANGE_TIMEOUT {
                                    warn!(%stream_id, "screencast GPU completion stalled; retaining buffer until its fence signals");
                                    warned = true;
                                }
                                delay = (delay * 2).min(MAX_FENCE_POLL_DELAY);
                                TimeoutAction::ToDuration(delay)
                            } else {
                                TimeoutAction::Drop
                            }
                        })
                        .unwrap();
                }
            }
            Some(sync_fd) => {
                trace!("scheduling buffer to queue");
                let stream_id = self.stream_id;
                let source = Generic::new(sync_fd, Interest::READ, Mode::OneShot);
                self.event_loop
                    .insert_source(source, move |_, _, state| {
                        for cast in &mut state.niri.casting.casts {
                            if cast.stream_id == stream_id {
                                cast.queue_completed_buffers();
                            }
                        }

                        Ok(PostAction::Remove)
                    })
                    .unwrap();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dequeue_buffer_and_render<R: NiriCaptureRenderer>(
        &mut self,
        renderer: &mut R,
        mut elements: &[CastRenderElement<R>],
        cursor_data: &CursorData<CastRenderElement<R>>,
        size: Size<i32, Physical>,
        scale: Scale<f64>,
        reference_luminance: f64,
    ) -> bool
    where
        R::Error: Send + Sync + 'static,
        CastRenderElement<R>: RenderElement<R>,
    {
        let mut inner = self.inner.borrow_mut();

        let parameters_changed =
            inner
                .render_cache
                .prepare(renderer, size, scale, reference_luminance);

        let CastState::Ready {
            damage_tracker,
            cursor_damage_tracker,
            pending_frame,
            ..
        } = &mut inner.state
        else {
            error!("cast must be in Ready state to render");
            return false;
        };
        let damage_tracker = damage_tracker
            .get_or_insert_with(|| OutputDamageTracker::new(size, scale, Transform::Normal));
        let cursor_damage_tracker = cursor_damage_tracker.get_or_insert_with(|| {
            OutputDamageTracker::new(
                Size::from((CURSOR_WIDTH as _, CURSOR_HEIGHT as _)),
                scale,
                Transform::Normal,
            )
        });

        // Size change will drop the damage tracker, but scale change won't, so check it here.
        let OutputModeSource::Static { scale: t_scale, .. } = damage_tracker.mode() else {
            unreachable!();
        };
        if *t_scale != scale {
            *damage_tracker = OutputDamageTracker::new(size, scale, Transform::Normal);
            *cursor_damage_tracker = OutputDamageTracker::new(
                Size::from((CURSOR_WIDTH as _, CURSOR_HEIGHT as _)),
                scale,
                Transform::Normal,
            );
        }

        let mut cursor_damaged = false;

        // For embedded cursor, pass the full slice (cursor + main) to the damage tracker.
        // For metadata or hidden cursor, pass only the main elements.
        if self.cursor_mode == CursorMode::Metadata || self.cursor_mode == CursorMode::Hidden {
            elements = &elements[cursor_data.elem_count..];
        }
        let (damage, _states) = damage_tracker.damage_output(1, elements).unwrap();

        if self.cursor_mode == CursorMode::Metadata {
            let (damage, _states) = cursor_damage_tracker
                .damage_output(1, &cursor_data.relocated)
                .unwrap();
            cursor_damaged = damage.is_some();
        }

        let cursor_location =
            (self.cursor_mode == CursorMode::Metadata).then_some(cursor_data.location);
        if !pending_frame.update(
            damage.is_some() || parameters_changed,
            cursor_damaged,
            cursor_location,
        ) {
            trace!("no damage, skipping frame");
            return false;
        }
        let redraw_cursor = pending_frame.cursor_damaged;
        drop(inner);

        let Some(pw_buffer) = self.dequeue_available_buffer() else {
            warn!("no available buffer in pw stream, skipping frame");
            return false;
        };
        let buffer = pw_buffer.as_ptr();

        let mut inner = self.inner.borrow_mut();
        let inner_ = &mut *inner;
        let CastState::Ready { alpha, .. } = &mut inner_.state else {
            unreachable!()
        };
        let alpha = *alpha;

        unsafe {
            let spa_buffer = (*buffer).buffer;

            let cursor_updated = self.cursor_mode != CursorMode::Metadata
                || add_cursor_metadata(renderer, spa_buffer, cursor_data, redraw_cursor);

            // Every submitted buffer still contains a complete image, including for consumers
            // that cannot handle metadata-only frames. Reused buffers only redraw their damage.
            let fd = (*(*spa_buffer).datas).fd;

            let res = match (*(*spa_buffer).datas).type_ {
                x if x == DataType::DmaBuf.as_raw() => {
                    let dmabuf = inner_.dmabufs[&fd].clone();
                    inner_
                        .render_cache
                        .render_dmabuf(renderer, fd, dmabuf, elements)
                        .map(|x| (x, SharingBuf::Dma))
                }
                x if x == DataType::MemFd.as_raw() => {
                    let shmbuf = &inner_.shmbufs[&fd];

                    let fourcc = if alpha {
                        Fourcc::Argb8888
                    } else {
                        Fourcc::Xrgb8888
                    };

                    inner_
                        .render_cache
                        .render_shm(renderer, fd, shmbuf, fourcc, elements)
                        .map(|()| (SyncPoint::signaled(), SharingBuf::Shm(shmbuf.layout)))
                }
                _ => Err(anyhow::anyhow!(
                    "unknown data type in dequeue_buffer_and_render"
                )),
            };

            if res.is_ok() {
                let CastState::Ready { pending_frame, .. } = &mut inner.state else {
                    unreachable!()
                };
                pending_frame.submitted(cursor_location, cursor_updated);
            }
            drop(inner);
            if res.is_ok() {
                if cursor_updated {
                    if let Some(token) = self.cursor_retry.take() {
                        self.event_loop.remove(token);
                    }
                } else {
                    self.schedule_cursor_retry();
                }
            }
            match res {
                Ok((sync_point, buf)) => {
                    if self.sequence_counter == 0 {
                        debug!(%self.stream_id, "rendered first screencast frame");
                    }
                    mark_buffer_as_good(pw_buffer, &mut self.sequence_counter, buf);
                    trace!("queueing buffer with seq={}", self.sequence_counter);
                    self.queue_after_sync(pw_buffer, sync_point);
                    true
                }
                Err(err) => {
                    warn!("error rendering to buffer: {err:?}");
                    if (*(*spa_buffer).datas).type_ == DataType::DmaBuf.as_raw() {
                        let _ = self.to_niri.send(PwToNiri::FallbackToShm {
                            stream_id: self.stream_id,
                            retry_dma: true,
                        });
                    }
                    return_unused_buffer(&self.stream, pw_buffer);
                    false
                }
            }
        }
    }

    pub fn dequeue_buffer_and_clear<R: NiriCaptureRenderer>(&mut self, renderer: &mut R) -> bool
    where
        R::Error: Send + Sync + 'static,
    {
        let mut inner = self.inner.borrow_mut();

        if self.device_change.is_some() || !matches!(inner.state, CastState::Ready { .. }) {
            return false;
        }

        inner.reset_damage();
        drop(inner);

        let Some(pw_buffer) = self.dequeue_available_buffer() else {
            warn!("no available buffer in pw stream, skipping frame");
            return false;
        };
        let buffer = pw_buffer.as_ptr();

        unsafe {
            let spa_buffer = (*buffer).buffer;

            if self.cursor_mode == CursorMode::Metadata {
                add_invisible_cursor(spa_buffer);
            }

            let fd = (*(*spa_buffer).datas).fd;

            let res = match (*(*(*buffer).buffer).datas).type_ {
                x if x == DataType::DmaBuf.as_raw() => {
                    let dmabuf = self.inner.borrow().dmabufs[&fd].clone();
                    clear_dmabuf(renderer, dmabuf).map(|x| (x, SharingBuf::Dma))
                }
                x if x == DataType::MemFd.as_raw() => {
                    let inner = self.inner.borrow();
                    let shmbuf = &inner.shmbufs[&fd];
                    clear_shmbuf(shmbuf);
                    Ok((SyncPoint::signaled(), SharingBuf::Shm(shmbuf.layout)))
                }
                _ => Err(anyhow::anyhow!(
                    "unknown data type in dequeue_buffer_and_clear"
                )),
            };

            match res {
                Ok((sync_point, buf)) => {
                    mark_buffer_as_good(pw_buffer, &mut self.sequence_counter, buf);
                    trace!("queueing clear buffer with seq={}", self.sequence_counter);
                    self.queue_after_sync(pw_buffer, sync_point);
                    true
                }
                Err(err) => {
                    warn!("error clearing buffer: {err:?}");
                    if (*(*spa_buffer).datas).type_ == DataType::DmaBuf.as_raw() {
                        let _ = self.to_niri.send(PwToNiri::FallbackToShm {
                            stream_id: self.stream_id,
                            retry_dma: true,
                        });
                    }
                    return_unused_buffer(&self.stream, pw_buffer);
                    false
                }
            }
        }
    }
}

impl Drop for Cast {
    fn drop(&mut self) {
        for token in [
            self.scheduled_redraw.take(),
            self.cursor_retry.take(),
            self.device_change_watchdog.take(),
            self.dma_retry.take(),
        ]
        .into_iter()
        .flatten()
        {
            self.event_loop.remove(token);
        }
    }
}

impl CastInner {
    fn reset_damage(&mut self) {
        self.state.reset_damage();
        self.render_cache = RenderCache::default();
    }

    unsafe fn on_add_buffer(
        &mut self,
        gbm: Option<&GbmDevice<DeviceFd>>,
        buffer: *mut pw_buffer,
    ) -> anyhow::Result<bool> {
        let CastState::Ready {
            size,
            alpha,
            dma_negotiation,
            ..
        } = self.state
        else {
            trace!("pw stream: add_buffer, but not ready yet");
            return Ok(false);
        };

        match dma_negotiation {
            Some(DmaNegotiation { modifier, .. }) => {
                trace!(
                    "pw stream: add_buffer (dma), size={size:?}, \
                     alpha={alpha}, modifier={modifier:?}"
                );

                let Some(gbm) = gbm else {
                    error!("add_buffer(dma) without gbm");
                    bail!("missing gbm");
                };

                unsafe {
                    let spa_buffer = (*buffer).buffer;

                    let fourcc = if alpha {
                        Fourcc::Argb8888
                    } else {
                        Fourcc::Xrgb8888
                    };

                    let dmabuf = allocate_dmabuf(gbm, size, fourcc, modifier)
                        .context("error allocating dmabuf")?;

                    let plane_count = dmabuf.num_planes();
                    assert_eq!((*spa_buffer).n_datas as usize, plane_count);
                    debug!(?size, %fourcc, ?modifier, plane_count, "allocated screencast DMA-BUF");

                    for (i, (fd, (stride, offset))) in
                        zip(dmabuf.handles(), zip(dmabuf.strides(), dmabuf.offsets())).enumerate()
                    {
                        let spa_data = (*spa_buffer).datas.add(i);
                        assert!((*spa_data).type_ & (1 << DataType::DmaBuf.as_raw()) > 0);

                        (*spa_data).type_ = DataType::DmaBuf.as_raw();

                        // With DMA-BUFs, consumers should ignore the maxsize field, and
                        // producers are allowed to set it to 0.
                        //
                        // https://docs.pipewire.org/page_dma_buf.html
                        (*spa_data).maxsize = 1;
                        (*spa_data).fd = fd.as_raw_fd() as i64;
                        (*spa_data).flags = SPA_DATA_FLAG_READWRITE;

                        let chunk = (*spa_data).chunk;
                        (*chunk).stride = stride as i32;
                        (*chunk).offset = offset;

                        trace!(
                            "pw buffer plane: fd={}, stride={stride}, offset={offset}",
                            (*spa_data).fd
                        );
                    }

                    let fd = (*(*spa_buffer).datas).fd;
                    assert!(self.dmabufs.insert(fd, dmabuf).is_none());
                }

                let first = self.dmabufs.len() == 1;
                if first {
                    self.reset_damage();
                }
                Ok(first)
            }
            None => {
                trace!("pw stream: add_buffer (shm), size={size:?}, alpha={alpha}");
                unsafe {
                    let spa_buffer = (*buffer).buffer;

                    let shmbuf = allocate_shmbuf(size).context("error allocating shmbuf")?;

                    assert_eq!((*spa_buffer).n_datas as usize, SHM_BLOCKS);

                    let spa_data = (*spa_buffer).datas;
                    assert!((*spa_data).type_ & (1 << DataType::MemFd.as_raw()) > 0);

                    (*spa_data).type_ = DataType::MemFd.as_raw();
                    (*spa_data).maxsize = shmbuf.layout.size;
                    (*spa_data).fd = shmbuf.fd.as_raw_fd() as i64;
                    (*spa_data).flags = SPA_DATA_FLAG_READWRITE;

                    let chunk = (*spa_data).chunk;
                    (*chunk).stride = shmbuf.layout.stride;
                    (*chunk).offset = 0;

                    let fd = (*(*spa_buffer).datas).fd;
                    assert!(self.shmbufs.insert(fd, shmbuf).is_none());
                }

                let first = self.shmbufs.len() == 1;
                if first {
                    self.reset_damage();
                }
                Ok(first)
            }
        }
    }

    unsafe fn on_remove_buffer(&mut self, buffer: *mut pw_buffer) {
        self.rendering_buffers
            .retain(|(buf, _)| buf.as_ptr() != buffer);

        unsafe {
            let spa_buffer = (*buffer).buffer;
            let spa_data = (*spa_buffer).datas;

            if (*spa_data).type_ == DataType::DmaBuf.as_raw() {
                trace!("pw stream: remove_buffer (dma)");
                assert!((*spa_buffer).n_datas > 0);

                let fd = (*spa_data).fd;
                self.render_cache.remove_buffer(fd);
                self.dmabufs.remove(&fd);
            } else if (*spa_data).type_ == DataType::MemFd.as_raw() {
                trace!("pw stream: remove_buffer (shm)");
                assert_eq!((*spa_buffer).n_datas, SHM_BLOCKS as u32);

                let fd = (*spa_data).fd;
                self.render_cache.remove_buffer(fd);
                self.shmbufs.remove(&fd);
            } else {
                error!(
                    "pw stream: remove_buffer (unknown type): {:?}",
                    (*spa_data).type_
                );
            }
        }
    }
}

impl CastState {
    fn reset_damage(&mut self) {
        if let Self::Ready {
            damage_tracker,
            cursor_damage_tracker,
            pending_frame,
            ..
        } = self
        {
            *damage_tracker = None;
            *cursor_damage_tracker = None;
            *pending_frame = PendingFrame::default();
        }
    }

    fn pending_size(&self) -> Option<Size<u32, Physical>> {
        match self {
            CastState::ResizePending { pending_size } => Some(*pending_size),
            CastState::ConfirmationPending { size, .. } => Some(*size),
            CastState::Ready { .. } => None,
        }
    }

    fn expected_format_size(&self) -> Size<u32, Physical> {
        match self {
            CastState::ResizePending { pending_size } => *pending_size,
            CastState::ConfirmationPending { size, .. } => *size,
            CastState::Ready { size, .. } => *size,
        }
    }
}

fn pw_version_supports_cursor_metadata() -> bool {
    // This PipeWire version fixed a critical memory issue with cursor metadata:
    // https://gitlab.freedesktop.org/pipewire/pipewire/-/merge_requests/2538
    unsafe { pw_check_library_version(1, 4, 8) }
}

fn make_pod(buffer: &mut Vec<u8>, object: pod::Object) -> &Pod {
    PodSerializer::serialize(Cursor::new(&mut *buffer), &pod::Value::Object(object)).unwrap();
    Pod::from_bytes(buffer).unwrap()
}

pub(super) fn test_implicit_buffer<R: NiriCaptureRenderer>(
    renderer: &mut R,
    gbm: &GbmDevice<DeviceFd>,
    size: Size<i32, Physical>,
) -> anyhow::Result<()>
where
    R::Error: Send + Sync + 'static,
{
    ensure!(size.w > 0 && size.h > 0, "invalid screencast buffer size");
    let bo = gbm
        .create_buffer_object::<()>(
            size.w as u32,
            size.h as u32,
            Fourcc::Xrgb8888,
            GbmBufferFlags::RENDERING,
        )
        .context("error allocating implicit screencast test buffer")?;
    // Test exactly the metadata that would be shared with an implicit consumer.
    // A successful import using GBM's private explicit layout is not sufficient.
    let buffer = GbmBuffer::from_bo(bo, true);
    let mut dmabuf = buffer
        .export()
        .context("error exporting implicit screencast test buffer")?;
    let _target = renderer
        .bind(&mut dmabuf)
        .context("error binding implicit screencast test buffer")?;
    Ok(())
}

fn parse_modifier_candidates(pod: &Pod) -> anyhow::Result<Vec<i64>> {
    let (_, value) = PodDeserializer::deserialize_from::<pod::Value>(pod.as_bytes())
        .map_err(|err| anyhow::anyhow!("error parsing modifier property: {err:?}"))?;
    debug!(?value, "negotiated modifier property");

    let modifiers = match value {
        // SPA can reduce an intersection to Choice_None while keeping DONT_FIXATE.
        // A scalar Long is also a fixed value in SPA. Both still need test allocation,
        // including DRM_FORMAT_MOD_INVALID, which requests an implicit layout.
        pod::Value::Long(modifier)
        | pod::Value::Choice(ChoiceValue::Long(Choice(_, ChoiceEnum::None(modifier)))) => {
            vec![modifier]
        }
        pod::Value::Choice(ChoiceValue::Long(Choice(_, ChoiceEnum::Enum { alternatives, .. }))) => {
            alternatives
        }
        _ => bail!("unexpected modifier property: {value:?}"),
    };

    ensure!(!modifiers.is_empty(), "empty modifier candidate list");
    Ok(modifiers)
}

fn find_preferred_modifier(
    gbm: &GbmDevice<DeviceFd>,
    size: Size<u32, Physical>,
    fourcc: Fourcc,
    modifiers: Vec<i64>,
) -> anyhow::Result<(Modifier, usize)> {
    debug!("find_preferred_modifier: size={size:?}, fourcc={fourcc}, modifiers={modifiers:?}");

    let (buffer, modifier) = allocate_buffer(gbm, size, fourcc, &modifiers)?;

    let dmabuf = buffer
        .export()
        .context("error exporting GBM buffer object as dmabuf")?;
    let plane_count = dmabuf.num_planes();

    // FIXME: Ideally this also needs to try binding the dmabuf for rendering.

    Ok((modifier, plane_count))
}

fn allocate_buffer(
    gbm: &GbmDevice<DeviceFd>,
    size: Size<u32, Physical>,
    fourcc: Fourcc,
    modifiers: &[i64],
) -> anyhow::Result<(GbmBuffer, Modifier)> {
    let (w, h) = (size.w, size.h);
    let flags = GbmBufferFlags::RENDERING;

    if modifiers.len() == 1 && Modifier::from(modifiers[0] as u64) == Modifier::Invalid {
        let bo = gbm
            .create_buffer_object::<()>(w, h, fourcc, flags)
            .context("error creating GBM buffer object")?;

        let buffer = GbmBuffer::from_bo(bo, true);
        Ok((buffer, Modifier::Invalid))
    } else {
        let modifiers = modifiers
            .iter()
            .map(|m| Modifier::from(*m as u64))
            .filter(|m| *m != Modifier::Invalid);

        let bo = gbm
            .create_buffer_object_with_modifiers2::<()>(w, h, fourcc, modifiers, flags)
            .context("error creating GBM buffer object")?;

        let modifier = bo.modifier();
        let buffer = GbmBuffer::from_bo(bo, false);
        Ok((buffer, modifier))
    }
}

fn allocate_dmabuf(
    gbm: &GbmDevice<DeviceFd>,
    size: Size<u32, Physical>,
    fourcc: Fourcc,
    modifier: Modifier,
) -> anyhow::Result<Dmabuf> {
    let (buffer, _modifier) = allocate_buffer(gbm, size, fourcc, &[u64::from(modifier) as i64])?;
    let dmabuf = buffer
        .export()
        .context("error exporting GBM buffer object as dmabuf")?;
    Ok(dmabuf)
}

#[derive(Debug)]
pub struct Shmbuf {
    fd: OwnedFd,
    layout: ShmLayout,
    mapping: ShmMapping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShmLayout {
    stride: i32,
    size: u32,
}

impl ShmLayout {
    fn new(size: Size<u32, Physical>) -> anyhow::Result<Self> {
        let stride = size
            .w
            .checked_mul(SHM_BYTES_PER_PIXEL as u32)
            .context("SHM stride overflows u32")?;
        let buffer_size = stride
            .checked_mul(size.h)
            .context("SHM buffer size overflows u32")?;

        Ok(Self {
            stride: stride.try_into().context("SHM stride exceeds i32")?,
            size: buffer_size,
        })
    }

    fn size_usize(self) -> usize {
        self.size as usize
    }
}

enum SharingBuf {
    Dma,
    Shm(ShmLayout),
}

fn allocate_shmbuf(size: Size<u32, Physical>) -> anyhow::Result<Shmbuf> {
    let layout = ShmLayout::new(size)?;
    let fd = memfd_create(
        "niri-pw-stream-memfd",
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .context("error creating memfd")?;
    ftruncate(&fd, layout.size.into()).context("error setting size of the fd")?;
    fcntl_add_seals(&fd, SealFlags::SEAL | SealFlags::SHRINK | SealFlags::GROW)
        .context("error sealing the fd")?;
    // SAFETY: The file has the requested size and is sealed against shrinking.
    // We only access the mapping while we own the dequeued PipeWire buffer.
    let mapping = unsafe { ShmMapping::new(fd.as_fd(), layout.size_usize()) }?;
    Ok(Shmbuf {
        fd,
        layout,
        mapping,
    })
}

unsafe fn return_unused_buffer(stream: &Stream, pw_buffer: NonNull<pw_buffer>) {
    // pw_stream_return_buffer() requires too new PipeWire (1.4.0). So, mark as
    // corrupted and queue.
    let pw_buffer = pw_buffer.as_ptr();
    let spa_buffer = (*pw_buffer).buffer;
    let chunk = (*(*spa_buffer).datas).chunk;
    // Some (older?) consumers will check for size == 0 instead of the CORRUPTED flag.
    (*chunk).size = 0;
    (*chunk).flags = SPA_CHUNK_FLAG_CORRUPTED as i32;

    if let Some(header) = find_meta_header(spa_buffer) {
        let header = header.as_ptr();
        (*header).flags = SPA_META_HEADER_FLAG_CORRUPTED;
    }

    pw_stream_queue_buffer(stream.as_raw_ptr(), pw_buffer);
}

unsafe fn mark_buffer_as_good(pw_buffer: NonNull<pw_buffer>, sequence: &mut u64, buf: SharingBuf) {
    let pw_buffer = pw_buffer.as_ptr();
    let spa_buffer = (*pw_buffer).buffer;
    let chunk = (*(*spa_buffer).datas).chunk;

    match buf {
        SharingBuf::Dma => {
            // With DMA-BUFs, consumers should ignore the size field, and producers are allowed
            // to set it to 0.
            //
            // https://docs.pipewire.org/page_dma_buf.html
            //
            // However, OBS checks for size != 0 as a workaround for old compositor versions,
            // so we set it to 1.
            (*chunk).size = 1;
            // Clear the corrupted flag we may have set before.
            (*chunk).flags = SPA_CHUNK_FLAG_NONE as i32;
        }
        SharingBuf::Shm(layout) => {
            (*chunk).size = layout.size;
            (*chunk).flags = SPA_CHUNK_FLAG_NONE as i32;
        }
    }

    *sequence = sequence.wrapping_add(1);
    if let Some(header) = find_meta_header(spa_buffer) {
        let header = header.as_ptr();
        // Clear the corrupted flag we may have set before.
        (*header).flags = 0;
        (*header).seq = *sequence;

        // Set buffer timestamp as unknown.
        //
        // FIXME: we could try passing real presentation timestamps for rendered frames here.
        // However, then we must also ensure that the time base never jumps (e.g. when switching a
        // dynamic cast between outputs) as this would mess up the timing downstream.
        (*header).pts = -1;
    }
}

unsafe fn find_meta_header(buffer: *mut spa_buffer) -> Option<NonNull<spa_meta_header>> {
    let p = spa_buffer_find_meta_data(buffer, SPA_META_Header, size_of::<spa_meta_header>()).cast();
    NonNull::new(p)
}

unsafe fn add_invisible_cursor(spa_buffer: *mut spa_buffer) {
    unsafe {
        let cursor_meta_ptr: *mut spa_meta_cursor = spa_buffer_find_meta_data(
            spa_buffer,
            SPA_META_Cursor,
            mem::size_of::<spa_meta_cursor>(),
        )
        .cast();
        let Some(cursor_meta) = cursor_meta_ptr.as_mut() else {
            return;
        };

        // The cursor is present but invisible.
        cursor_meta.id = 1;
        cursor_meta.position.x = 0;
        cursor_meta.position.y = 0;
        cursor_meta.hotspot.x = 0;
        cursor_meta.hotspot.y = 0;
        cursor_meta.bitmap_offset = BITMAP_META_OFFSET as _;

        let bitmap_meta_ptr = cursor_meta_ptr
            .byte_add(BITMAP_META_OFFSET)
            .cast::<spa_meta_bitmap>();
        let bitmap_meta = &mut *bitmap_meta_ptr;

        // HACK: PipeWire docs say offset = 0 means invisible.
        //
        // Unfortunately, OBS doesn't actually check that, instead it checks that size isn't zero:
        // https://github.com/obsproject/obs-studio/blob/f4aaa5f0417c5ec40a3799551e125129fce1e007/plugins/linux-pipewire/pipewire.c#L900
        //
        // Unfortunately, libwebrtc, on top of ignoring offset, also treats size = 0 as "preserve
        // previous cursor":
        // https://webrtc.googlesource.com/src/+/97b46e12582606a238d4f0c8524365cf5bdcb411/modules/desktop_capture/linux/wayland/shared_screencast_stream.cc#765
        //
        // So, send a 1x1 transparent pixel instead...
        bitmap_meta.offset = BITMAP_DATA_OFFSET as _;
        bitmap_meta.size.width = 1;
        bitmap_meta.size.height = 1;
        bitmap_meta.stride = CURSOR_BPP as i32;
        bitmap_meta.format = CURSOR_FORMAT;

        let bitmap_data = bitmap_meta_ptr.cast::<u8>().add(BITMAP_DATA_OFFSET);
        let bitmap_slice = slice::from_raw_parts_mut(bitmap_data, CURSOR_BITMAP_SIZE);
        bitmap_slice[..4].copy_from_slice(&[0, 0, 0, 0]);
    }
}

unsafe fn add_cursor_metadata<R: NiriCaptureRenderer>(
    renderer: &mut R,
    spa_buffer: *mut spa_buffer,
    cursor_data: &CursorData<impl RenderElement<R>>,
    redraw: bool,
) -> bool {
    unsafe {
        let cursor_meta_ptr: *mut spa_meta_cursor = spa_buffer_find_meta_data(
            spa_buffer,
            SPA_META_Cursor,
            mem::size_of::<spa_meta_cursor>(),
        )
        .cast();
        let Some(cursor_meta) = cursor_meta_ptr.as_mut() else {
            return true;
        };

        cursor_meta.id = 1;
        cursor_meta.position.x = cursor_data.location.x;
        cursor_meta.position.y = cursor_data.location.y;
        cursor_meta.hotspot.x = cursor_data.hotspot.x;
        cursor_meta.hotspot.y = cursor_data.hotspot.y;

        if !redraw {
            trace!("cursor not damaged, skipping rerendering");
            cursor_meta.bitmap_offset = 0;
            return true;
        }

        cursor_meta.bitmap_offset = BITMAP_META_OFFSET as _;

        let bitmap_meta_ptr = cursor_meta_ptr
            .byte_add(BITMAP_META_OFFSET)
            .cast::<spa_meta_bitmap>();
        let bitmap_meta = &mut *bitmap_meta_ptr;

        // Start with a 1x1 transparent pixel; see comment in add_invisible_cursor().
        bitmap_meta.offset = BITMAP_DATA_OFFSET as _;
        bitmap_meta.size.width = 1;
        bitmap_meta.size.height = 1;
        bitmap_meta.stride = CURSOR_BPP as i32;
        bitmap_meta.format = CURSOR_FORMAT;

        let bitmap_data = bitmap_meta_ptr.cast::<u8>().add(BITMAP_DATA_OFFSET);
        let bitmap_slice = slice::from_raw_parts_mut(bitmap_data, CURSOR_BITMAP_SIZE);
        bitmap_slice[..4].copy_from_slice(&[0, 0, 0, 0]);

        let size = Size::new(
            min(cursor_data.size.w, CURSOR_WIDTH as i32),
            min(cursor_data.size.h, CURSOR_HEIGHT as i32),
        );
        if size.w == 0 || size.h == 0 {
            trace!("cursor is invisible, skipping rendering");
            return true;
        }

        let _span = tracy_client::span!("add_cursor_metadata render cursor");

        // FIXME: use a reliable buffer whenever we're rendering the cursor.
        //
        // PipeWire buffers are not normally guaranteed to reach the destination, so our buffer
        // with the rendered cursor bitmap may not reach the consumer.
        //
        // Reliable buffers should be available starting from 1.6.0:
        // https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/4885
        let mapping = match render_and_download(
            renderer,
            size,
            cursor_data.scale,
            Transform::Normal,
            Fourcc::Argb8888,
            cursor_data.relocated.iter().rev(),
        ) {
            Ok(mapping) => mapping,
            Err(err) => {
                warn!("error rendering cursor: {err:?}");
                return false;
            }
        };
        let pixels = match renderer.map_texture(&mapping) {
            Ok(pixels) => pixels,
            Err(err) => {
                warn!("error mapping cursor texture: {err:?}");
                return false;
            }
        };

        bitmap_slice[..pixels.len()].copy_from_slice(pixels);

        // Fill the metadata now that everything succeeded.
        bitmap_meta.size.width = size.w as _;
        bitmap_meta.size.height = size.h as _;
        bitmap_meta.stride = size.w * CURSOR_BPP as i32;
        true
    }
}

fn clear_shmbuf(buffer: &Shmbuf) {
    buffer.mapping.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_state(dma: bool) -> CastState {
        CastState::Ready {
            size: Size::from((3840, 2160)),
            alpha: true,
            dma_negotiation: dma.then_some(DmaNegotiation {
                modifier: Modifier::Invalid,
                plane_count: 1,
            }),
            damage_tracker: None,
            cursor_damage_tracker: None,
            pending_frame: PendingFrame::default(),
        }
    }

    #[test]
    fn device_migration_waits_for_old_gpu_and_buffer_retirement() {
        let device = CastDevice {
            node: None,
            gbm: None,
        };
        let mut change = DeviceChange::WaitingForFrames(device);
        let dma = ready_state(true);
        // A consumer has not returned the buffers yet, and the GPU still writes one frame.
        assert_eq!(change.next_step(1, 8, true, &dma), DeviceChangeStep::Wait);
        assert_eq!(
            change.next_step(0, 8, true, &dma),
            DeviceChangeStep::NegotiateShm
        );
        change = DeviceChange::WaitingForShm(change.into_device());
        // Matching dimensions and modifier never suffice to reuse the old GPU's buffers.
        assert_eq!(change.next_step(0, 8, false, &dma), DeviceChangeStep::Wait);
        let shm = ready_state(false);
        assert_eq!(change.next_step(0, 1, false, &shm), DeviceChangeStep::Wait);
        assert_eq!(
            change.next_step(0, 0, false, &shm),
            DeviceChangeStep::Install
        );
    }

    #[test]
    fn device_migration_requires_shm_confirmation_even_without_old_buffers() {
        let change = DeviceChange::WaitingForShm(CastDevice {
            node: None,
            gbm: None,
        });
        let pending = CastState::ResizePending {
            pending_size: Size::from((3840, 2160)),
        };
        assert_eq!(
            change.next_step(0, 0, false, &pending),
            DeviceChangeStep::Wait
        );
        assert_eq!(
            change.next_step(0, 0, false, &ready_state(true)),
            DeviceChangeStep::Wait
        );
        assert_eq!(
            change.next_step(0, 0, false, &ready_state(false)),
            DeviceChangeStep::Install
        );
    }

    #[test]
    fn device_migration_from_shm_still_drains_in_flight_frames() {
        let change = DeviceChange::WaitingForFrames(CastDevice {
            node: None,
            gbm: None,
        });
        let shm = ready_state(false);
        assert_eq!(change.next_step(1, 0, false, &shm), DeviceChangeStep::Wait);
        assert_eq!(
            change.next_step(0, 0, false, &shm),
            DeviceChangeStep::Install
        );
    }

    #[test]
    fn device_migration_can_reuse_shm_when_consumer_rejects_dma() {
        let change = DeviceChange::WaitingForFrames(CastDevice {
            node: None,
            gbm: None,
        });
        let mut state = ready_state(false);
        // An allocator can be available even though the consumer negotiated SHM.
        assert_eq!(
            change.next_step(0, 0, true, &state),
            DeviceChangeStep::Install
        );
        state.reset_damage();
        assert!(matches!(
            state,
            CastState::Ready {
                dma_negotiation: None,
                ..
            }
        ));
    }

    #[test]
    fn backpressure_preserves_final_content_and_cursor_bitmap() {
        let mut frame = PendingFrame::default();
        let location = Some(Point::from((50, 60)));
        frame.submitted(location, true);

        // The damage trackers see a single change, but the consumer owns all buffers.
        assert!(frame.update(true, true, location));
        // Subsequent identical frames have no new damage. The pending update must still render,
        // including the cursor bitmap, once a buffer becomes available.
        for _ in 0..3 {
            assert!(frame.update(false, false, location));
            assert!(frame.cursor_damaged);
        }
        // Main content can succeed while downloading the cursor bitmap fails.
        frame.submitted(location, false);
        assert!(frame.update(false, false, location));
        assert!(frame.cursor_damaged);
        frame.submitted(location, true);
        assert!(!frame.update(false, false, location));
    }

    #[test]
    fn backpressure_preserves_final_cursor_movement() {
        let mut frame = PendingFrame::default();
        let initial = Some(Point::from((50, 60)));
        let moved = Some(Point::from((80, 90)));
        frame.submitted(initial, true);
        assert!(frame.update(false, false, moved));
        assert!(frame.update(false, false, moved));
        assert_eq!(frame.last_cursor_location, initial);
        assert!(!frame.cursor_damaged);
        frame.submitted(moved, true);
        assert!(!frame.update(false, false, moved));
    }

    fn modifier_candidates(value: pod::Value) -> anyhow::Result<Vec<i64>> {
        let mut object =
            make_video_params(VideoFormat::BGRx, &[], Size::from((3840, 2160)), 144_000);
        object.properties.push(Property {
            key: FormatProperties::VideoModifier.as_raw(),
            flags: PropertyFlags::MANDATORY | PropertyFlags::DONT_FIXATE,
            value,
        });
        let mut bytes = Vec::new();
        let pod = make_pod(&mut bytes, object);
        let object = pod.as_object().unwrap();
        let property = object
            .find_prop(spa::utils::Id(FormatProperties::VideoModifier.0))
            .unwrap();
        assert!(property.flags().contains(PodPropFlags::DONT_FIXATE));
        parse_modifier_candidates(property.value())
    }

    #[test]
    fn modifier_candidates_accept_single_values() {
        // In particular, do not turn the implicit-layout marker into Linear or SHM.
        for modifier in [0, 0x00ff_ffff_ffff_ffff, 0x0300_0000_0060_6010] {
            let value = pod::Value::Choice(ChoiceValue::Long(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::None(modifier),
            )));
            assert_eq!(modifier_candidates(value).unwrap(), vec![modifier]);
            assert_eq!(
                modifier_candidates(pod::Value::Long(modifier)).unwrap(),
                vec![modifier]
            );
        }
    }

    #[test]
    fn modifier_candidates_accept_spa_filtered_pod() {
        // SPA's filter produces this POD when two Enum lists intersect only at
        // DRM_FORMAT_MOD_INVALID, even with DONT_FIXATE on both properties.
        // Build the native-endian wire data independently of PodSerializer.
        let header = [
            24_u32,
            SPA_TYPE_Choice,
            SPA_CHOICE_None,
            0,
            8,
            SPA_TYPE_Long,
        ];
        let bytes: Vec<u8> = header
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .chain(0x00ff_ffff_ffff_ffff_i64.to_ne_bytes())
            .collect();
        let pod = Pod::from_bytes(&bytes).unwrap();
        assert_eq!(
            parse_modifier_candidates(pod).unwrap(),
            vec![0x00ff_ffff_ffff_ffff]
        );
    }

    #[test]
    fn modifier_candidates_preserve_enum_alternatives() {
        let alternatives = vec![0x0300_0000_0060_6010, 0, 0x00ff_ffff_ffff_ffff];
        let value = pod::Value::Choice(ChoiceValue::Long(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Enum {
                // The default is a preference, not an additional candidate.
                default: 42,
                alternatives: alternatives.clone(),
            },
        )));
        assert_eq!(modifier_candidates(value).unwrap(), alternatives);
    }

    #[test]
    fn modifier_candidates_reject_invalid_types_and_choices() {
        assert!(modifier_candidates(pod::Value::Int(0)).is_err());
        assert!(
            modifier_candidates(pod::Value::Choice(ChoiceValue::Int(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::None(0),
            ))))
            .is_err()
        );

        for choice in [
            ChoiceEnum::Enum {
                default: 0,
                alternatives: vec![],
            },
            ChoiceEnum::Range {
                default: 0,
                min: 0,
                max: 1,
            },
            ChoiceEnum::Step {
                default: 0,
                min: 0,
                max: 1,
                step: 1,
            },
            ChoiceEnum::Flags {
                default: 0,
                flags: vec![0],
            },
        ] {
            let value = pod::Value::Choice(ChoiceValue::Long(Choice(ChoiceFlags::empty(), choice)));
            assert!(modifier_candidates(value).is_err());
        }
    }

    #[test]
    fn shm_layout_uses_spa_representable_dimensions() {
        let layout = ShmLayout::new(Size::from((3840, 2160))).unwrap();
        assert_eq!(layout.stride, 15360);
        assert_eq!(layout.size, 33_177_600);

        assert!(ShmLayout::new(Size::from((536_870_912, 1))).is_err());
        assert!(ShmLayout::new(Size::from((500_000_000, 3))).is_err());
    }
}
