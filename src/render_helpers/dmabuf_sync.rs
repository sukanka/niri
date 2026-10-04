//! Publish compositor reads to clients that reuse DMA-BUFs through implicit synchronization.

use std::collections::HashSet;
use std::io;
use std::os::fd::{AsFd, BorrowedFd};
use std::sync::atomic::{AtomicBool, Ordering};

use smithay::backend::allocator::dmabuf::{import_sync_file, Dmabuf, SyncFileFlags};
use smithay::backend::renderer::sync::SyncPoint;

/// Adds the completion fence of a GPU read to every plane's implicit reservation object.
///
/// Exporting a rendering fence alone does not prevent an implicit-sync client from
/// overwriting its buffer. A READ fence makes the client's next write wait for this read.
/// Multi-plane buffers may have distinct reservation objects, so import into every plane.
pub fn import_read_fence(dmabuf: &Dmabuf, fence: BorrowedFd<'_>) -> io::Result<()> {
    for plane in dmabuf.handles() {
        import_sync_file(plane, SyncFileFlags::READ, fence)?;
    }
    Ok(())
}

/// DMA-BUFs that already received the read fence for one submission.
///
/// Cloned handles have the same identity, while independently imported DMA-BUFs are kept
/// distinct even if their metadata matches. This only deduplicates implicit imports: each
/// surface buffer must still receive its own explicit release fence.
#[derive(Default)]
pub struct ReadFenceImports {
    imported: HashSet<Dmabuf>,
}

impl ReadFenceImports {
    pub fn import(&mut self, dmabuf: &Dmabuf, fence: BorrowedFd<'_>) -> io::Result<()> {
        self.import_with(dmabuf, || import_read_fence(dmabuf, fence))
    }

    fn import_with(
        &mut self,
        dmabuf: &Dmabuf,
        import: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        if self.imported.contains(dmabuf) {
            return Ok(());
        }

        import()?;
        self.imported.insert(dmabuf.clone());
        Ok(())
    }
}

/// Exports and publishes a read fence, waiting for completion if either step is unavailable.
///
/// Call before releasing any buffer used by this submission. `publish` may attach the same
/// fence to explicit release points as well as implicit DMA-BUF reservations. An error after
/// a partial publication is safe: the fallback waits for the entire submission to complete.
pub fn with_read_fence(sync: &SyncPoint, publish: impl FnOnce(BorrowedFd<'_>) -> io::Result<()>) {
    // Once the GPU read is complete, the client can already reuse the buffers. In particular,
    // callers that had to wait before queueing do not need to export or publish another fence.
    if sync.is_reached() {
        return;
    }

    if let Some(fence) = sync.export() {
        match publish(fence.as_fd()) {
            Ok(()) => return,
            Err(err) => {
                static LOGGED_IMPORT_FAILURE: AtomicBool = AtomicBool::new(false);
                if !LOGGED_IMPORT_FAILURE.swap(true, Ordering::Relaxed) {
                    debug!("cannot publish DMA-BUF read fence, waiting for GPU completion: {err}");
                }
            }
        }
    }

    wait_for_read_completion(sync);
}

/// Interrupted waits must be retried: interruption gives no completion guarantee.
pub fn wait_for_read_completion(sync: &SyncPoint) {
    while sync.wait().is_err() {}
}

#[cfg(test)]
mod tests {
    use std::os::fd::OwnedFd;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use smithay::backend::allocator::dmabuf::DmabufFlags;
    use smithay::backend::allocator::{Fourcc, Modifier};
    use smithay::backend::renderer::sync::{Fence, Interrupted};

    use super::*;

    #[derive(Debug)]
    struct TestFence {
        waits: Arc<AtomicUsize>,
        exports: AtomicUsize,
        interruptions: usize,
        exportable: bool,
    }

    impl Fence for TestFence {
        fn is_signaled(&self) -> bool {
            self.waits.load(Ordering::Relaxed) > self.interruptions
        }

        fn wait(&self) -> Result<(), Interrupted> {
            if self.waits.fetch_add(1, Ordering::Relaxed) < self.interruptions {
                Err(Interrupted)
            } else {
                Ok(())
            }
        }

        fn is_exportable(&self) -> bool {
            self.exportable
        }

        fn export(&self) -> Option<OwnedFd> {
            self.exports.fetch_add(1, Ordering::Relaxed);
            self.exportable
                .then(|| std::fs::File::open("/dev/null").unwrap().into())
        }
    }

    fn fence(exportable: bool, interruptions: usize) -> (SyncPoint, Arc<AtomicUsize>) {
        let waits = Arc::new(AtomicUsize::new(0));
        let sync = TestFence {
            waits: Arc::clone(&waits),
            exports: AtomicUsize::new(0),
            interruptions,
            exportable,
        }
        .into();
        (sync, waits)
    }

    fn dmabuf() -> Dmabuf {
        // Only handle identity is used: no DMA-BUF ioctl is issued by these tests.
        let fd: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        let mut builder = Dmabuf::builder(
            (16, 16),
            Fourcc::Abgr8888,
            Modifier::Linear,
            DmabufFlags::empty(),
        );
        assert!(builder.add_plane(fd, 0, 64));
        builder.build().unwrap()
    }

    #[test]
    fn completed_read_needs_no_export_publication_or_wait() {
        for exportable in [false, true] {
            let (sync, waits) = fence(exportable, 0);
            sync.wait().unwrap();
            with_read_fence(&sync, |_| panic!("the GPU read already completed"));
            assert_eq!(waits.load(Ordering::Relaxed), 1);
            assert_eq!(
                sync.get::<TestFence>()
                    .unwrap()
                    .exports
                    .load(Ordering::Relaxed),
                0
            );
        }
    }

    #[test]
    fn read_fence_imports_deduplicate_aliases_only_within_one_submission() {
        let buffer = dmabuf();
        let alias = buffer.clone();
        let other = dmabuf();
        let mut imported = ReadFenceImports::default();
        let mut imports = 0;
        for buffer in [&buffer, &alias, &other] {
            imported
                .import_with(buffer, || {
                    imports += 1;
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(imports, 2);

        let mut next_submission = ReadFenceImports::default();
        next_submission
            .import_with(&alias, || {
                imports += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(imports, 3);
    }

    #[test]
    fn failed_read_fence_import_does_not_mark_buffer_published() {
        let buffer = dmabuf();
        let mut imported = ReadFenceImports::default();
        assert!(imported
            .import_with(&buffer, || {
                Err(io::Error::from_raw_os_error(libc::ENOTTY))
            })
            .is_err());

        let mut retried = false;
        imported
            .import_with(&buffer, || {
                retried = true;
                Ok(())
            })
            .unwrap();
        assert!(retried);
    }

    #[test]
    fn unexportable_read_waits_through_interruptions() {
        let (sync, waits) = fence(false, 2);
        with_read_fence(&sync, |_| panic!("a fence was not exported"));
        assert_eq!(waits.load(Ordering::Relaxed), 3);
        assert!(sync.is_reached());
    }

    #[test]
    fn rejected_read_fence_waits_before_returning() {
        let (sync, waits) = fence(true, 1);
        let mut attempted = false;
        with_read_fence(&sync, |_| {
            attempted = true;
            Err(io::Error::from_raw_os_error(libc::ENOTTY))
        });
        assert!(attempted);
        assert_eq!(waits.load(Ordering::Relaxed), 2);
        assert!(sync.is_reached());
    }

    #[test]
    fn published_read_fence_keeps_submission_asynchronous() {
        let (sync, waits) = fence(true, 0);
        let mut published = false;
        with_read_fence(&sync, |_| {
            published = true;
            Ok(())
        });
        assert!(published);
        assert_eq!(waits.load(Ordering::Relaxed), 0);
    }
}
