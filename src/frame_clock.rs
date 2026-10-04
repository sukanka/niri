use std::num::NonZeroU64;
use std::time::Duration;

use crate::utils::get_monotonic_time;

#[derive(Debug)]
pub struct FrameClock {
    last_presentation_time: Option<Duration>,
    refresh_interval_ns: Option<NonZeroU64>,
    vrr: bool,
    // CPU time through render preparation and submission. This is a scheduling
    // estimate, not a GPU duration or a reason to postpone a ready frame.
    render_budget: Duration,
}

impl FrameClock {
    pub fn new(refresh_interval: Option<Duration>, vrr: bool) -> Self {
        let refresh_interval_ns = if let Some(interval) = &refresh_interval {
            assert_eq!(interval.as_secs(), 0);
            Some(NonZeroU64::new(interval.subsec_nanos().into()).unwrap())
        } else {
            None
        };

        Self {
            last_presentation_time: None,
            refresh_interval_ns,
            vrr,
            render_budget: Duration::from_millis(1),
        }
    }

    pub fn refresh_interval(&self) -> Option<Duration> {
        self.refresh_interval_ns
            .map(|r| Duration::from_nanos(r.get()))
    }

    pub fn set_vrr(&mut self, vrr: bool) {
        if self.vrr == vrr {
            return;
        }

        self.vrr = vrr;
        self.last_presentation_time = None;
        self.render_budget = Duration::from_millis(1);
    }

    pub fn vrr(&self) -> bool {
        self.vrr
    }

    pub fn presented(&mut self, presentation_time: Duration) {
        if presentation_time.is_zero() {
            // Not interested in these.
            return;
        }

        self.last_presentation_time = Some(presentation_time);
    }

    pub fn next_presentation_time(&self) -> Duration {
        self.next_presentation_time_at(get_monotonic_time())
    }

    /// Latest estimated start for an already queued output. Use a shared `now`
    /// when comparing outputs so sorting cannot move a deadline across a vblank.
    pub fn render_start_deadline(&self, now: Duration) -> Duration {
        self.next_presentation_time_at(now)
            .saturating_sub(self.render_budget)
    }

    pub fn record_render_duration(&mut self, elapsed: Duration) {
        // Follow expensive frames immediately, then decay slowly so one cheap
        // direct-scanout frame does not erase the budget for a composited frame.
        // Bound outliers (device recovery, debugger pauses) to one refresh or 16ms.
        let limit = self.refresh_interval().unwrap_or(Duration::from_millis(16));
        let sample = elapsed.clamp(Duration::from_micros(100).min(limit), limit);
        let decayed = self.render_budget.mul_f64(0.95);
        self.render_budget = sample.max(decayed).min(limit);
    }

    fn next_presentation_time_at(&self, mut now: Duration) -> Duration {
        let Some(refresh_interval_ns) = self.refresh_interval_ns else {
            return now;
        };
        let Some(last_presentation_time) = self.last_presentation_time else {
            return now;
        };

        let refresh_interval_ns = refresh_interval_ns.get();

        if now <= last_presentation_time {
            // Got an early VBlank.
            let orig_now = now;
            now += Duration::from_nanos(refresh_interval_ns);

            if now < last_presentation_time {
                // Not sure when this can happen.
                error!(
                    now = ?orig_now,
                    ?last_presentation_time,
                    "got a 2+ early VBlank, {:?} until presentation",
                    last_presentation_time - now,
                );
                now = last_presentation_time + Duration::from_nanos(refresh_interval_ns);
            }
        }

        let since_last = now - last_presentation_time;
        let since_last_ns =
            since_last.as_secs() * 1_000_000_000 + u64::from(since_last.subsec_nanos());
        let to_next_ns = (since_last_ns / refresh_interval_ns + 1) * refresh_interval_ns;

        // If VRR is enabled and more than one frame passed since last presentation, assume that we
        // can present immediately.
        if self.vrr && to_next_ns > refresh_interval_ns {
            now
        } else {
            last_presentation_time + Duration::from_nanos(to_next_ns)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_deadlines_prioritize_earlier_refresh_and_expensive_outputs() {
        let now = Duration::from_millis(101);
        let mut slow = FrameClock::new(Some(Duration::from_millis(16)), false);
        let mut fast = FrameClock::new(Some(Duration::from_millis(4)), false);
        slow.presented(Duration::from_millis(100));
        fast.presented(Duration::from_millis(100));
        assert!(fast.render_start_deadline(now) < slow.render_start_deadline(now));
        let mut expensive = FrameClock::new(Some(Duration::from_millis(16)), false);
        expensive.presented(Duration::from_millis(100));
        expensive.record_render_duration(Duration::from_millis(5));
        assert!(expensive.render_start_deadline(now) < slow.render_start_deadline(now));
        // Budgeting only orders work: presentation and frame callback timing stay unchanged.
        assert_eq!(
            expensive.next_presentation_time_at(now),
            slow.next_presentation_time_at(now)
        );
    }

    #[test]
    fn render_budget_is_bounded_and_recovers_from_outliers() {
        let mut clock = FrameClock::new(Some(Duration::from_millis(4)), false);
        clock.record_render_duration(Duration::from_secs(20));
        assert_eq!(clock.render_budget, Duration::from_millis(4));
        for _ in 0..200 {
            clock.record_render_duration(Duration::ZERO);
        }
        assert_eq!(clock.render_budget, Duration::from_micros(100));
        clock.set_vrr(true);
        assert_eq!(clock.render_budget, Duration::from_millis(1));
        clock.presented(Duration::from_millis(100));
        let now = Duration::from_millis(110);
        assert!(clock.render_start_deadline(now) <= now);
        assert_eq!(clock.next_presentation_time_at(now), now);
        let unknown = FrameClock::new(None, false);
        assert!(unknown.render_start_deadline(now) <= now);
    }
}
