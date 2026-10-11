//! Bounded recovery for NVIDIA's stale per-CRTC VSIF cache after connector detach.
//!
//! A standard neutral HDMI VSIF must reach a real commit before clearing it can
//! refresh the driver's cache. TEST_ONLY and deferred frames cannot advance this
//! sequence. All intermediate pixels and signalling remain SDR.

use smithay::backend::drm::{Colorspace, ConnectorColorState, Eotf, HdmiVsifMetadata};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Prime {
        neutral: ConnectorColorState,
        requested: ConnectorColorState,
    },
    Retry {
        staged: Option<ConnectorColorState>,
    },
    Clear(ConnectorColorState),
    Blocked(ConnectorColorState),
}

#[derive(Debug, Default)]
pub(super) struct HdrVsifRecovery {
    attempted: bool,
    phase: Phase,
}

impl HdrVsifRecovery {
    /// The budget is per surface/device reset, not per redraw or config reload.
    pub(super) fn begin(
        &mut self,
        supported: bool,
        current: ConnectorColorState,
        requested: ConnectorColorState,
    ) -> bool {
        if self.attempted
            || self.phase != Phase::Idle
            || !supported
            || current.colorspace != Colorspace::Default
            || current.hdr_metadata.is_some()
            || current.hdmi_vsif.is_some()
            || requested.hdmi_vsif.is_some()
            || !requested
                .hdr_metadata
                .is_some_and(|metadata| metadata.eotf == Eotf::SmpteSt2084)
        {
            return false;
        }
        self.attempted = true;
        self.phase = Phase::Prime {
            neutral: ConnectorColorState {
                hdmi_vsif: Some(HdmiVsifMetadata::hdmi_no_additional_video_format()),
                ..current
            },
            requested,
        };
        true
    }

    /// Re-evaluate the latest configured target after the neutral commit, so
    /// turning HDR off or selecting LLDV during recovery remains effective.
    pub(super) fn target(&self, requested: ConnectorColorState) -> Option<ConnectorColorState> {
        match self.phase {
            Phase::Idle | Phase::Retry { .. } => Some(requested),
            Phase::Prime { neutral, .. } => Some(neutral),
            Phase::Clear(state) => Some(state),
            Phase::Blocked(_) => None,
        }
    }

    pub(super) fn staged(&mut self, target: ConnectorColorState) {
        if let Phase::Retry { staged } = &mut self.phase {
            *staged = Some(target);
        }
    }

    /// Observe accepted DRM state. Matching pending alone is never sufficient.
    /// Returns whether another redraw is needed to finish the transition.
    pub(super) fn applied(
        &mut self,
        current: ConnectorColorState,
        pending: ConnectorColorState,
    ) -> bool {
        if current == pending {
            match self.phase {
                Phase::Prime { neutral, .. } if current == neutral => {
                    self.phase = Phase::Retry { staged: None };
                }
                Phase::Retry {
                    staged: Some(target),
                }
                | Phase::Clear(target)
                    if current == target =>
                {
                    self.phase = Phase::Idle;
                }
                _ => (),
            }
        }
        self.in_progress()
    }

    /// A permanent failure of either TEST_ONLY or a real frame cancels the
    /// staged transition. Return the original failed request to suppress until
    /// configuration changes, while clearing any temporary VSIF with SDR pixels.
    pub(super) fn failed(
        &mut self,
        target: ConnectorColorState,
        current: ConnectorColorState,
    ) -> Option<ConnectorColorState> {
        let failed = match self.phase {
            Phase::Prime { requested, .. } => requested,
            Phase::Retry { .. } => target,
            Phase::Clear(state) => {
                self.phase = Phase::Blocked(state);
                return None;
            }
            Phase::Idle | Phase::Blocked(_) => return None,
        };
        self.phase = Phase::Clear(ConnectorColorState {
            max_bpc: current.max_bpc,
            ..Default::default()
        });
        Some(failed)
    }

    pub(super) fn in_progress(&self) -> bool {
        matches!(
            self.phase,
            Phase::Prime { .. } | Phase::Retry { .. } | Phase::Clear(_)
        )
    }

    pub(super) fn clearing(&self) -> bool {
        matches!(self.phase, Phase::Clear(_) | Phase::Blocked(_))
    }

    /// Allow an explicit config change to retry failed cleanup, without opening
    /// another neutral/HDR cycle. Active transitions keep their ordering.
    pub(super) fn config_changed(&mut self) {
        if let Phase::Blocked(state) = self.phase {
            self.phase = Phase::Clear(state);
        }
    }

    /// Call only after Smithay discarded old queued frames during device reset.
    pub(super) fn device_reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use smithay::backend::drm::HdrOutputMetadata;

    use super::*;

    fn hdr10() -> ConnectorColorState {
        ConnectorColorState {
            colorspace: Colorspace::Bt2020Rgb,
            hdr_metadata: Some(HdrOutputMetadata::pq_bt2020(554, 1, 554, 315)),
            max_bpc: Some(12),
            ..Default::default()
        }
    }

    #[test]
    fn only_real_commits_advance_recovery() {
        let sdr = ConnectorColorState {
            max_bpc: Some(10),
            ..Default::default()
        };
        let hdr = hdr10();
        let mut recovery = HdrVsifRecovery::default();
        assert!(recovery.begin(true, sdr, hdr));
        let neutral = recovery.target(hdr).unwrap();
        assert_eq!(neutral.colorspace, Colorspace::Default);
        assert!(neutral.hdr_metadata.is_none());
        assert_eq!(neutral.max_bpc, Some(10));
        assert_eq!(neutral.hdmi_vsif.unwrap().as_bytes(), &[3, 12, 0, 0]);
        recovery.staged(neutral);
        assert!(recovery.applied(sdr, neutral)); // TEST_ONLY or deferred queue.
        assert_eq!(recovery.target(hdr), Some(neutral));
        assert!(recovery.applied(neutral, neutral)); // Schedule HDR redraw.
        assert_eq!(recovery.target(hdr), Some(hdr));
        recovery.staged(hdr);
        assert!(recovery.applied(neutral, hdr)); // HDR not yet committed.
        assert!(!recovery.applied(hdr, hdr));
        assert!(!recovery.begin(true, sdr, hdr)); // Budget remains consumed.
    }

    #[test]
    fn latest_request_wins_after_neutral_commit() {
        let sdr = ConnectorColorState::default();
        let ll_dv = ConnectorColorState {
            colorspace: Colorspace::Bt2020Rgb,
            hdmi_vsif: Some(HdmiVsifMetadata::low_latency_dolby_vision()),
            ..Default::default()
        };
        for latest in [
            sdr,
            ll_dv,
            ConnectorColorState {
                max_bpc: Some(10),
                ..hdr10()
            },
        ] {
            let mut recovery = HdrVsifRecovery::default();
            assert!(recovery.begin(true, sdr, hdr10()));
            let neutral = recovery.target(hdr10()).unwrap();
            recovery.config_changed();
            assert_eq!(recovery.target(latest), Some(neutral));
            recovery.applied(neutral, neutral);
            assert_eq!(recovery.target(latest), Some(latest));
            recovery.staged(latest);
            assert!(!recovery.applied(latest, latest));
        }
    }

    #[test]
    fn failed_prime_or_hdr_commit_clears_and_stops() {
        let sdr = ConnectorColorState::default();
        let hdr = hdr10();
        for prime_committed in [false, true] {
            let mut recovery = HdrVsifRecovery::default();
            assert!(recovery.begin(true, sdr, hdr));
            let neutral = recovery.target(hdr).unwrap();
            let (current, pending) = if prime_committed {
                recovery.applied(neutral, neutral);
                recovery.staged(hdr);
                (neutral, hdr)
            } else {
                (sdr, neutral)
            };
            assert_eq!(recovery.failed(pending, current), Some(hdr));
            assert_eq!(recovery.target(hdr), Some(sdr));
            // Even if current is already SDR, pending must be cancelled first.
            assert!(recovery.applied(current, pending));
            assert!(!recovery.applied(sdr, sdr));
            recovery.config_changed();
            assert!(!recovery.begin(true, sdr, hdr));
        }
    }

    #[test]
    fn permanent_cleanup_failure_is_bounded_and_config_can_retry_cleanup() {
        let sdr = ConnectorColorState::default();
        let mut recovery = HdrVsifRecovery::default();
        recovery.begin(true, sdr, hdr10());
        let neutral = recovery.target(hdr10()).unwrap();
        recovery.applied(neutral, neutral);
        recovery.failed(hdr10(), neutral);
        assert_eq!(recovery.failed(sdr, neutral), None);
        assert_eq!(recovery.target(hdr10()), None);
        assert!(!recovery.in_progress());
        recovery.config_changed();
        assert_eq!(recovery.target(hdr10()), Some(sdr));
        assert!(!recovery.begin(true, sdr, hdr10()));
    }

    #[test]
    fn reset_discards_transition_and_starts_a_new_device_epoch() {
        let sdr = ConnectorColorState::default();
        let mut recovery = HdrVsifRecovery::default();
        recovery.begin(true, sdr, hdr10());
        recovery.device_reset();
        assert_eq!(recovery.target(hdr10()), Some(hdr10()));
        assert!(recovery.begin(true, sdr, hdr10()));
    }

    #[test]
    fn recovery_does_not_disrupt_an_active_hdr_transport() {
        let sdr = ConnectorColorState::default();
        let mut recovery = HdrVsifRecovery::default();
        assert!(!recovery.begin(false, sdr, hdr10()));
        assert!(!recovery.begin(true, hdr10(), hdr10()));
        let ll_dv = ConnectorColorState {
            hdmi_vsif: Some(HdmiVsifMetadata::low_latency_dolby_vision()),
            ..sdr
        };
        assert!(!recovery.begin(true, ll_dv, hdr10()));
        assert!(!recovery.begin(true, sdr, ll_dv));
        assert!(!recovery.begin(true, sdr, sdr));
        assert!(recovery.begin(true, sdr, hdr10()));
    }
}
