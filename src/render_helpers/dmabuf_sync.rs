//! Publish compositor reads to clients that reuse DMA-BUFs through implicit synchronization.

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

/// Exports and publishes a read fence, waiting for completion if either step is unavailable.
///
/// Call before releasing any buffer used by this submission. `publish` may attach the same
/// fence to explicit release points as well as implicit DMA-BUF reservations. An error after
/// a partial publication is safe: the fallback waits for the entire submission to complete.
pub fn with_read_fence(sync: &SyncPoint, publish: impl FnOnce(BorrowedFd<'_>) -> io::Result<()>) {
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

    use smithay::backend::renderer::sync::{Fence, Interrupted};

    use super::*;

    #[derive(Debug)]
    struct TestFence {
        waits: Arc<AtomicUsize>,
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
            self.exportable
                .then(|| std::fs::File::open("/dev/null").unwrap().into())
        }
    }

    fn fence(exportable: bool, interruptions: usize) -> (SyncPoint, Arc<AtomicUsize>) {
        let waits = Arc::new(AtomicUsize::new(0));
        let sync = TestFence {
            waits: Arc::clone(&waits),
            interruptions,
            exportable,
        }
        .into();
        (sync, waits)
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
