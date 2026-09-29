//! Opt-in hardware diagnostic: no window, DRM master, modeset, or real session lock.
//!
//! Run only on a machine with two render nodes. Source/target default to the AMD/NVIDIA
//! nodes on the development laptop; override NIRI_MULTIGPU_SOURCE/TARGET if needed.
//! NIRI_MULTIGPU_NATIVE=1 chooses the source GPU's native modifiers instead of linear.
//! NIRI_MULTIGPU_DIAGNOSTIC=1 also runs intentionally unsynchronized negative controls.
//! Unset __EGL_VENDOR_LIBRARY_FILENAMES and software-rendering overrides when running.

use std::fs::OpenOptions;
use std::os::fd::OwnedFd;
use std::time::Instant;

use anyhow::{ensure, Context as _};
use smithay::backend::allocator::dmabuf::{AsDmabuf, Dmabuf};
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Allocator, Buffer as _, Fourcc, Modifier};
use smithay::backend::drm::{DrmNode, NodeType};
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::gles::{ffi, GlesRenderer, GlesTexture};
use smithay::backend::renderer::multigpu::gbm::GbmGlesBackend;
use smithay::backend::renderer::multigpu::GpuManager;
use smithay::backend::renderer::{
    Bind, Color32F, ExportMem, Frame, ImportDmaWl, Offscreen, Renderer,
};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::Resource as _;
use smithay::utils::{Buffer, DeviceFd, Physical, Rectangle, Transform};
use smithay::wayland::compositor::with_states;

use super::Fixture;
use crate::render_helpers::dmabuf_sync::{
    import_read_fence, wait_for_read_completion, with_read_fence,
};

type Api = GbmGlesBackend<GlesRenderer, DeviceFd>;

#[derive(Clone, Copy, Debug)]
enum ImportPath {
    Unspecified,
    SourceHint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyncAction {
    None,
    ExportOnly,
    ReadFence,
    Wait,
}

#[derive(Debug)]
struct Outcome {
    path: ImportPath,
    sync_action: SyncAction,
    serial: bool,
    mismatched_frames: usize,
    mismatched_pixels: usize,
    direct_imports: usize,
    published_read_fences: usize,
    elapsed_ms: u128,
}

fn parameter(name: &str, default: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .clamp(1, max)
}

fn open_render_node(path: &str) -> anyhow::Result<(DrmNode, GbmDevice<DeviceFd>)> {
    let expected = DrmNode::from_path(path)?;
    ensure!(
        expected.ty() == NodeType::Render,
        "test only opens render nodes"
    );
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let node = DrmNode::from_file(&file)?;
    ensure!(node == expected, "DRM node changed while opening it");
    let fd: OwnedFd = file.into();
    Ok((node, GbmDevice::new(DeviceFd::from(fd))?))
}

fn renderer_name(renderer: &mut GlesRenderer) -> anyhow::Result<String> {
    Ok(renderer.with_context(|gl| unsafe {
        std::ffi::CStr::from_ptr(gl.GetString(ffi::RENDERER).cast())
            .to_string_lossy()
            .into_owned()
    })?)
}

fn source_renderer(gbm: GbmDevice<DeviceFd>) -> anyhow::Result<GlesRenderer> {
    // This models the client's independent context, separate from both compositor contexts.
    unsafe {
        let display = EGLDisplay::new(gbm)?;
        let context = EGLContext::new(&display)?;
        Ok(GlesRenderer::new(context)?)
    }
}

fn dummy_surface() -> (Fixture, WlSurface) {
    let mut fixture = Fixture::new();
    fixture.add_output(1, (256, 256));
    let client = fixture.add_client();
    let window = fixture.client(client).create_window();
    let client_surface = window.surface.clone();
    window.commit();
    fixture.roundtrip(client);
    let window = fixture.client(client).window(&client_surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    fixture.double_roundtrip(client);
    let surface = fixture
        .niri()
        .layout
        .windows()
        .next()
        .unwrap()
        .1
        .toplevel()
        .wl_surface()
        .clone();
    (fixture, surface)
}

fn color(index: usize) -> [u8; 4] {
    // Four colors with three source slots ensure a reused client buffer changes contents.
    match index % 4 {
        0 => [255, 0, 0, 255],
        1 => [0, 255, 0, 255],
        2 => [0, 0, 255, 255],
        _ => [255, 255, 0, 255],
    }
}

fn fill_source(
    renderer: &mut GlesRenderer,
    buffer: &mut Dmabuf,
    size: i32,
    index: usize,
) -> anyhow::Result<()> {
    let top = color(index);
    let top = Color32F::new(
        f32::from(top[0]) / 255.,
        f32::from(top[1]) / 255.,
        f32::from(top[2]) / 255.,
        1.,
    );
    let mut target = renderer.bind(buffer)?;
    let mut frame = renderer.render(&mut target, (size, size).into(), Transform::Normal)?;
    frame.clear(
        Color32F::new(1., 0., 1., 1.),
        &[Rectangle::from_size((size, size).into())],
    )?;
    frame.clear(
        top,
        &[Rectangle::new((0, 0).into(), (size, size / 2).into())],
    )?;
    // Match niri's pre-commit acquire blocker: the producer is finished before sampling.
    // This does not wait for earlier consumer reads unless implicit reservation fences work.
    frame.finish()?.wait()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    producer: &mut GlesRenderer,
    allocator: &mut GbmAllocator<DeviceFd>,
    gpus: &mut GpuManager<Api>,
    source_node: DrmNode,
    target_node: DrmNode,
    path: ImportPath,
    sync_action: SyncAction,
    serial: bool,
) -> anyhow::Result<Outcome> {
    let size = parameter("NIRI_MULTIGPU_SIZE", 256, 512) as i32;
    let frames = if serial {
        4
    } else {
        parameter("NIRI_MULTIGPU_FRAMES", 32, 64)
    };
    let repeats = if serial {
        1
    } else {
        parameter("NIRI_MULTIGPU_REPEATS", 64, 128)
    };
    let batch_size = if serial {
        1
    } else {
        parameter("NIRI_MULTIGPU_BATCH", 8, 16)
    };
    let full: Rectangle<i32, Physical> = Rectangle::from_size((size, size).into());
    let top_damage: Rectangle<i32, Buffer> = Rectangle::new((0, 0).into(), (size, size / 2).into());
    let native = std::env::var_os("NIRI_MULTIGPU_NATIVE").is_some();
    let modifiers = if native {
        Bind::<Dmabuf>::supported_formats(producer)
            .unwrap_or_default()
            .iter()
            .filter(|format| format.code == Fourcc::Abgr8888)
            .map(|format| format.modifier)
            .collect::<Vec<_>>()
    } else {
        vec![Modifier::Linear]
    };
    let mut buffers = (0..3)
        .map(|_| {
            allocator
                .create_buffer(size as u32, size as u32, Fourcc::Abgr8888, &modifiers)
                .context("allocate AMD source buffer")?
                .export()
                .context("export AMD source buffer")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    for buffer in &buffers {
        buffer.set_node(match path {
            ImportPath::Unspecified => None,
            ImportPath::SourceHint => Some(source_node),
        });
    }
    eprintln!(
        "case={path:?} sync={sync_action:?} serial={serial} source_format={:?}",
        buffers[0].format()
    );

    // The real SurfaceData makes import_dma_buffer reuse MultiTexture's DMA shadow slot,
    // including across rotating wl_buffers. ImportDma::import_dmabuf would allocate afresh.
    let (mut fixture, surface) = dummy_surface();
    let display = fixture.niri().display_handle.clone();
    let client = surface.client().unwrap();
    let wl_buffers = buffers
        .iter()
        .map(|buffer| {
            client.create_resource::<WlBuffer, Dmabuf, crate::niri::State>(
                &display,
                1,
                buffer.clone(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let start = Instant::now();
    let mut outcome = Outcome {
        path,
        sync_action,
        serial,
        mismatched_frames: 0,
        mismatched_pixels: 0,
        direct_imports: 0,
        published_read_fences: 0,
        elapsed_ms: 0,
    };
    let mut pending: Vec<(usize, GlesTexture)> = Vec::new();
    for index in 0..frames {
        let slot = index % buffers.len();
        fill_source(producer, &mut buffers[slot], size, index)?;
        let mut consumer = gpus.single_renderer(&target_node)?;
        let damage = if index == 0 {
            Rectangle::from_size((size, size).into())
        } else {
            top_damage
        };
        let texture = with_states(&surface, |states| {
            consumer.import_dma_buffer(&wl_buffers[slot], Some(states), &[damage])
        })?;
        outcome.direct_imports += usize::from(buffers[slot].node() == Some(target_node));
        if index < 3 {
            eprintln!("import case={path:?} serial={serial} slot={slot} selected={:?} source={source_node} target={target_node}", buffers[slot].node());
        }

        let mut target: GlesTexture =
            consumer.create_buffer(Fourcc::Abgr8888, (size, size).into())?;
        let sync = {
            let mut framebuffer = consumer.bind(&mut target)?;
            let mut frame =
                consumer.render(&mut framebuffer, (size, size).into(), Transform::Normal)?;
            frame.clear(Color32F::TRANSPARENT, &[full])?;
            // Bounded overdraw increases overlap with the next AMD update without a busy
            // shader, unbounded GPU loop, global stall, or any display/KMS operation.
            for _ in 0..repeats {
                frame.render_texture_from_to(
                    &texture,
                    Rectangle::from_size((size, size).into()).to_f64(),
                    full,
                    &[full],
                    &[],
                    Transform::Normal,
                    1.,
                )?;
            }
            frame.finish()?
        };
        match sync_action {
            SyncAction::None => {}
            SyncAction::ExportOnly => {
                let _fence = sync
                    .export()
                    .context("NVIDIA renderer cannot export native fence")?;
            }
            SyncAction::ReadFence => with_read_fence(&sync, |fence| {
                import_read_fence(&buffers[slot], fence)?;
                outcome.published_read_fences += 1;
                Ok(())
            }),
            SyncAction::Wait => wait_for_read_completion(&sync),
        }
        pending.push((index, target));
        if pending.len() == batch_size || index + 1 == frames {
            let raw: &mut GlesRenderer = consumer.as_mut();
            for (frame_index, target) in pending.drain(..) {
                let mapping = raw.copy_texture(
                    &target,
                    Rectangle::from_size((size, size).into()),
                    Fourcc::Abgr8888,
                )?;
                let pixels = raw.map_texture(&mapping)?;
                let mut mismatched = 0;
                let mut first = None;
                for (offset, pixel) in pixels.chunks_exact(4).enumerate() {
                    let expected = if offset / (size as usize) < (size / 2) as usize {
                        color(frame_index)
                    } else {
                        [255, 0, 255, 255]
                    };
                    if pixel != expected {
                        mismatched += 1;
                        first.get_or_insert((
                            offset % size as usize,
                            offset / size as usize,
                            pixel.to_vec(),
                            expected,
                        ));
                    }
                }
                if mismatched != 0 {
                    outcome.mismatched_frames += 1;
                    outcome.mismatched_pixels += mismatched;
                    if outcome.mismatched_frames <= 4 {
                        eprintln!("case={path:?} sync={sync_action:?} frame={frame_index} mismatch_pixels={mismatched} first={first:?}");
                    }
                }
            }
        }
    }
    outcome.elapsed_ms = start.elapsed().as_millis();
    eprintln!("OUTCOME serial={} {outcome:?}", outcome.serial);
    Ok(outcome)
}

#[test]
#[ignore = "manual AMD/NVIDIA render-node test; no display access; run with --nocapture"]
fn cross_gpu_client_buffer_reuse() -> anyhow::Result<()> {
    super::gpu::init_logging();
    let source =
        std::env::var("NIRI_MULTIGPU_SOURCE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    let target =
        std::env::var("NIRI_MULTIGPU_TARGET").unwrap_or_else(|_| "/dev/dri/renderD129".into());
    let (source_node, source_gbm) = open_render_node(&source)?;
    let (target_node, target_gbm) = open_render_node(&target)?;
    ensure!(source_node != target_node, "two distinct GPUs are required");
    let mut producer = source_renderer(source_gbm.clone())?;
    ensure!(
        !producer.is_software(),
        "source must be a hardware renderer"
    );
    eprintln!("SOURCE {source}: {}", renderer_name(&mut producer)?);
    let mut allocator = GbmAllocator::new(source_gbm.clone(), GbmBufferFlags::RENDERING);
    let mut api: Api = GbmGlesBackend::default();
    api.add_node(source_node, source_gbm)?;
    api.add_node(target_node, target_gbm)?;
    let mut gpus = GpuManager::new(api)?;
    {
        let mut renderer = gpus.single_renderer(&target_node)?;
        let raw: &mut GlesRenderer = renderer.as_mut();
        ensure!(!raw.is_software(), "target must be a hardware renderer");
        eprintln!("TARGET {target}: {}", renderer_name(raw)?);
    }
    let diagnostic = std::env::var_os("NIRI_MULTIGPU_DIAGNOSTIC").is_some();
    let mut failed = Vec::new();
    for path in [ImportPath::Unspecified, ImportPath::SourceHint] {
        let sanity = run_case(
            &mut producer,
            &mut allocator,
            &mut gpus,
            source_node,
            target_node,
            path,
            SyncAction::None,
            true,
        )?;
        ensure!(
            sanity.mismatched_frames == 0,
            "serial sanity failed; this is not evidence of a reuse race: {sanity:?}"
        );
        for sync_action in [
            SyncAction::None,
            SyncAction::ExportOnly,
            SyncAction::ReadFence,
            SyncAction::Wait,
        ] {
            // The shadow-only case also guards the lower-level consumer-fence handoff;
            // publishing a fence to the client buffer can incidentally mask that race.
            let shadow_regression =
                matches!(path, ImportPath::SourceHint) && sync_action == SyncAction::None;
            if !diagnostic && sync_action != SyncAction::ReadFence && !shadow_regression {
                continue;
            }
            let outcome = run_case(
                &mut producer,
                &mut allocator,
                &mut gpus,
                source_node,
                target_node,
                path,
                sync_action,
                false,
            )?;
            // Unsynchronized direct imports are a negative control, not production behavior.
            // DMA shadows must be safe even without the compositor's client-buffer bridge.
            let must_match = outcome.direct_imports == 0
                || matches!(sync_action, SyncAction::ReadFence | SyncAction::Wait);
            if must_match && outcome.mismatched_frames != 0 {
                failed.push((outcome.path, outcome.sync_action, outcome.mismatched_frames));
            }
        }
    }
    ensure!(failed.is_empty(), "cross-GPU frame corruption: {failed:?}");
    Ok(())
}
