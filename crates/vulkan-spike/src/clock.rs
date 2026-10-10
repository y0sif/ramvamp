//! Clock lock: an optional run mode that holds the CPU arm's cores and the
//! iGPU each at one fixed frequency for the whole run.
//!
//! Why it exists. At unlocked clocks neither unit holds still: the
//! 2026-10-09 runs had the CPU arm flip between two speeds for stretches of
//! 10-25 rounds, and the streamed GPU arm slow down for a third of the run,
//! in every run. A CPU and an iGPU in one package share a power and
//! frequency budget, and each unit's frequency governor moves its clock on
//! its own schedule, so a ratio between them depends on which state each
//! happened to be in. Locking both takes the governors out of the
//! comparison. The cost is that the two units are compared at chosen
//! clocks below their peak, not at the box's best burst. Prefill is
//! sustained work, so a sustained clock is the fairer operating point.
//!
//! `scripts/bench-unthrottled.sh --lock` sets the lock and passes the values
//! it set through the environment. This module does not trust that: it
//! checks the lock is in place before sampling, reads the clocks the units
//! actually ran at after every block, and a reading off the lock invalidates
//! the run. A lock that silently did not hold would otherwise print a clean
//! verdict about clocks nobody ran at.

use std::path::{Path, PathBuf};

/// How far an observed clock may sit from the lock, as a fraction of it.
/// Wide enough for the counters' own granularity (the iGPU steps in 50 MHz
/// bins), narrow enough to catch a unit the package has stepped down.
pub const TOL_LOCK: f64 = 0.05;

/// The environment variables `bench-unthrottled.sh --lock` sets.
pub const ENV_CPU_KHZ: &str = "RAMVAMP_SPIKE_LOCK_CPU_KHZ";
pub const ENV_GT_MHZ: &str = "RAMVAMP_SPIKE_LOCK_GT_MHZ";

/// The requested lock: one frequency for every pinned CPU, one for the iGPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lock {
    pub cpu_khz: u64,
    pub gt_mhz: u32,
}

impl Lock {
    /// How the lock reads in the verdict line.
    pub fn describe(&self, n_cpus: usize) -> String {
        format!(
            "locked clocks (CPU {:.2} GHz on {n_cpus} pinned cores, iGPU {} MHz)",
            self.cpu_khz as f64 / 1e6,
            self.gt_mhz
        )
    }
}

/// The lock requested through the environment, if any.
///
/// # Errors
///
/// When only one of the two variables is set, or either is not a positive
/// integer: a half-specified lock is a mistake, not an unlocked run.
pub fn requested() -> Result<Option<Lock>, String> {
    parse_lock(
        std::env::var(ENV_CPU_KHZ).ok().as_deref(),
        std::env::var(ENV_GT_MHZ).ok().as_deref(),
    )
}

fn parse_lock(cpu: Option<&str>, gt: Option<&str>) -> Result<Option<Lock>, String> {
    match (cpu, gt) {
        (None, None) => Ok(None),
        (Some(cpu), Some(gt)) => {
            let cpu_khz = cpu
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| format!("{ENV_CPU_KHZ}={cpu} is not a positive kHz value"))?;
            let gt_mhz = gt
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| format!("{ENV_GT_MHZ}={gt} is not a positive MHz value"))?;
            Ok(Some(Lock { cpu_khz, gt_mhz }))
        }
        _ => Err(format!(
            "only one of {ENV_CPU_KHZ} and {ENV_GT_MHZ} is set; a clock lock needs both"
        )),
    }
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// The render GT's frequency files.
///
/// On a part with more than one GT, such as Meteor Lake's render GT plus
/// media GT, the card-level `gt_*_freq_mhz` files are not the render GT's. A
/// write goes to every GT in turn and fails at the first that refuses it, and
/// a read aggregates across GTs: the lowest min, the highest max. On
/// 2026-10-10 a lock written through them set the render GT, was refused by
/// the media GT (maximum 1300 MHz), and read back half-applied. The restore
/// then wrote the media GT's 100 MHz minimum to a render GT whose floor is
/// 800, was refused, and left the render GT pinned at 1400. The shader runs
/// on the render GT, so these are its own files, `gt/gt0/rps_*_freq_mhz`.
/// The card-level files are only the fallback on a single-GT part, where
/// they are the same thing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gt {
    dir: PathBuf,
    prefix: &'static str,
}

impl Gt {
    /// One frequency file: `name` is `min`, `max`, `boost` or `act`.
    pub fn file(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{}{name}_freq_mhz", self.prefix))
    }

    pub fn describe(&self) -> String {
        self.dir.display().to_string()
    }

    /// The lowest clock the GT runs at while active. A reading under it is
    /// the GT idle or mid-transition, not a GT the package has slowed.
    pub fn rpn(&self) -> Option<u64> {
        read_u64(&self.file("RPn"))
    }
}

/// `cpus` plus every hyperthread sibling of each, sorted.
///
/// A lock has to cover the siblings. Two threads of one core share its
/// clock, and the core runs at the higher of their two requests, so an
/// unlocked sibling with work to do pulls a locked shard's core off the lock.
/// On 2026-10-10 the six locked shards were read at 2.26-4.50 GHz against a
/// 3.00 GHz lock while their siblings were left unlocked.
///
/// # Errors
///
/// When a CPU's `thread_siblings_list` cannot be read or parsed: the lock
/// would then be unchecked on a thread that can move the core's clock.
pub fn with_siblings(cpus: &[usize]) -> Result<Vec<usize>, String> {
    let mut all = Vec::new();
    for &c in cpus {
        let path = format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("could not read the hyperthread siblings of cpu{c}: {e}"))?;
        let list = parse_cpu_list(&text).ok_or_else(|| {
            format!("could not parse the hyperthread siblings of cpu{c}: {text:?}")
        })?;
        all.extend(list);
        all.push(c);
    }
    all.sort_unstable();
    all.dedup();
    Ok(all)
}

/// Parse a kernel CPU list such as `0,5` or `1-2,8`. `None` on anything
/// malformed, including an empty list and a descending range.
fn parse_cpu_list(text: &str) -> Option<Vec<usize>> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for part in text.split(',') {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?);
                if a > b {
                    return None;
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse().ok()?),
        }
    }
    Some(out)
}

/// The render GT of the one i915 card under `/sys/class/drm`. `None` when
/// there is no such card, or more than one and so no way to tell which is
/// the device Vulkan picked.
pub fn render_gt() -> Option<Gt> {
    find_render_gt(Path::new("/sys/class/drm"))
}

fn find_render_gt(drm: &Path) -> Option<Gt> {
    let mut found = std::fs::read_dir(drm)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card") && !n.contains('-'))
        })
        .filter_map(|card| {
            let gt0 = card.join("gt").join("gt0");
            if gt0.join("rps_min_freq_mhz").exists() {
                Some(Gt {
                    dir: gt0,
                    prefix: "rps_",
                })
            } else if card.join("gt_min_freq_mhz").exists() {
                Some(Gt {
                    dir: card,
                    prefix: "gt_",
                })
            } else {
                None
            }
        });
    let first = found.next()?;
    if found.next().is_some() {
        return None;
    }
    Some(first)
}

/// The lock settings as read back from sysfs, before any sampling.
pub struct Settings {
    /// `(cpu, scaling_min_freq, scaling_max_freq)` in kHz per pinned CPU.
    pub cpus: Vec<(usize, Option<u64>, Option<u64>)>,
    /// The render GT's `(min, max, boost)` in MHz, or `None` when there is
    /// no unambiguous i915 render GT.
    pub gt: Option<(Option<u64>, Option<u64>, Option<u64>)>,
}

/// Read the lock settings for `cpus` and the iGPU.
pub fn read_settings(cpus: &[usize], gt: Option<&Gt>) -> Settings {
    let cpu_attr = |cpu: usize, attr: &str| {
        read_u64(Path::new(&format!(
            "/sys/devices/system/cpu/cpu{cpu}/cpufreq/{attr}"
        )))
    };
    Settings {
        cpus: cpus
            .iter()
            .map(|&c| {
                (
                    c,
                    cpu_attr(c, "scaling_min_freq"),
                    cpu_attr(c, "scaling_max_freq"),
                )
            })
            .collect(),
        gt: gt.map(|g| {
            (
                read_u64(&g.file("min")),
                read_u64(&g.file("max")),
                read_u64(&g.file("boost")),
            )
        }),
    }
}

/// Every way the lock is not in place: a pinned CPU whose min and max are
/// not both the locked frequency, or an iGPU whose min, max and boost are
/// not all the locked frequency. Empty when the lock is set as requested.
pub fn check_in_place(lock: &Lock, settings: &Settings) -> Vec<String> {
    let mut problems = Vec::new();
    let off: Vec<String> = settings
        .cpus
        .iter()
        .filter(|(_, lo, hi)| *lo != Some(lock.cpu_khz) || *hi != Some(lock.cpu_khz))
        .map(|(c, lo, hi)| format!("cpu{c} min {} max {}", khz(*lo), khz(*hi)))
        .collect();
    if !off.is_empty() {
        problems.push(format!(
            "CPU not locked at {:.2} GHz: {}",
            lock.cpu_khz as f64 / 1e6,
            off.join(", ")
        ));
    }
    let want = Some(u64::from(lock.gt_mhz));
    match settings.gt {
        None => problems.push(
            "no single i915 render GT under /sys/class/drm, so the iGPU lock cannot be checked"
                .to_owned(),
        ),
        Some((lo, hi, boost)) if lo != want || hi != want || boost != want => {
            problems.push(format!(
                "iGPU not locked at {} MHz: min {} max {} boost {}",
                lock.gt_mhz,
                mhz(lo),
                mhz(hi),
                mhz(boost)
            ));
        }
        Some(_) => {}
    }
    problems
}

fn khz(v: Option<u64>) -> String {
    v.map_or_else(
        || "unreadable".to_owned(),
        |k| format!("{:.2} GHz", k as f64 / 1e6),
    )
}

fn mhz(v: Option<u64>) -> String {
    v.map_or_else(|| "unreadable".to_owned(), |m| format!("{m} MHz"))
}

/// The clocks the units actually ran at, read after each block of a run.
#[derive(Default)]
pub struct ClockLog {
    /// Every CPU reading as `(cpu, kHz)`.
    cpu: Vec<(usize, u64)>,
    cpu_unreadable: usize,
    /// Every iGPU reading at or above [`Self::gt_floor`], MHz.
    gt: Vec<u64>,
    /// The GT's lowest active clock (RPn). Readings under it are the GT
    /// idle or mid-transition; 0 when unknown, so only a zero counts.
    gt_floor: u64,
    /// iGPU readings of zero (idle in RC6), under the floor, or unreadable.
    gt_idle: usize,
}

impl ClockLog {
    /// A log that counts iGPU readings under `floor`, the GT's lowest active
    /// clock, as idle rather than as a slowed GT.
    pub fn with_gt_floor(floor: Option<u64>) -> Self {
        Self {
            gt_floor: floor.unwrap_or(0),
            ..Self::default()
        }
    }

    /// Read `scaling_cur_freq` on every pinned CPU. Called right after the
    /// CPU block, while the value still reflects the block.
    pub fn sample_cpus(&mut self, cpus: &[usize]) {
        for &c in cpus {
            let r = read_u64(Path::new(&format!(
                "/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"
            )));
            self.record_cpu(c, r);
        }
    }

    /// Read the render GT's actual frequency. Called between
    /// GPU iterations, never inside a timed one.
    pub fn sample_gt(&mut self, gt: Option<&Gt>) {
        let r = gt.and_then(|g| read_u64(&g.file("act")));
        self.record_gt(r);
    }

    fn record_cpu(&mut self, cpu: usize, khz: Option<u64>) {
        match khz {
            Some(k) => self.cpu.push((cpu, k)),
            None => self.cpu_unreadable += 1,
        }
    }

    fn record_gt(&mut self, mhz: Option<u64>) {
        match mhz {
            Some(m) if m > 0 && m >= self.gt_floor => self.gt.push(m),
            _ => self.gt_idle += 1,
        }
    }

    fn cpu_khz(&self) -> Vec<u64> {
        self.cpu.iter().map(|&(_, k)| k).collect()
    }

    fn range(xs: &[u64]) -> Option<(u64, u64)> {
        Some((*xs.iter().min()?, *xs.iter().max()?))
    }

    /// The `p` percentile of `xs` by nearest rank; `xs` must be non-empty.
    fn pct(xs: &[u64], p: f64) -> u64 {
        let mut s = xs.to_vec();
        s.sort_unstable();
        let rank = ((p * s.len() as f64).ceil() as usize).clamp(1, s.len());
        s[rank - 1]
    }

    /// One line per unit: the spread of clocks observed (5th, 50th and 95th
    /// percentile, and the full range), and how many readings were unusable.
    pub fn summary(&self) -> Vec<String> {
        let khz = self.cpu_khz();
        let cpu = match Self::range(&khz) {
            Some((lo, hi)) => format!(
                "CPU  observed p5/p50/p95 {:.2}/{:.2}/{:.2} GHz, range {:.2}-{:.2}, over {} readings",
                Self::pct(&khz, 0.05) as f64 / 1e6,
                Self::pct(&khz, 0.50) as f64 / 1e6,
                Self::pct(&khz, 0.95) as f64 / 1e6,
                lo as f64 / 1e6,
                hi as f64 / 1e6,
                khz.len()
            ),
            None => "CPU  never observed".to_owned(),
        };
        let gt = match Self::range(&self.gt) {
            Some((lo, hi)) => format!(
                "iGPU observed p5/p50/p95 {}/{}/{} MHz, range {lo}-{hi}, over {} readings",
                Self::pct(&self.gt, 0.05),
                Self::pct(&self.gt, 0.50),
                Self::pct(&self.gt, 0.95),
                self.gt.len()
            ),
            None => "iGPU never observed".to_owned(),
        };
        let floor = if self.gt_floor > 0 {
            format!(
                "zero, under the {} MHz lowest active clock, or unreadable",
                self.gt_floor
            )
        } else {
            "zero or unreadable".to_owned()
        };
        vec![
            format!("{cpu} ({} unreadable)", self.cpu_unreadable),
            format!("{gt} ({} idle: {floor})", self.gt_idle),
        ]
    }

    /// Every way the lock did not hold while the run was sampled: any
    /// reading outside [`TOL_LOCK`] of the lock, or a unit never observed
    /// at all, since a lock nobody saw hold is not evidence that it did.
    pub fn violations(&self, lock: &Lock) -> Vec<String> {
        let mut v = Vec::new();
        let off = |x: u64, want: u64| (x as f64 / want as f64 - 1.0).abs() > TOL_LOCK;
        match Self::range(&self.cpu_khz()) {
            None => v.push("CPU clock never observed".to_owned()),
            Some((lo, hi)) => {
                // Per CPU, so a single misbehaving core is told apart from
                // the whole package being stepped down.
                let mut per_cpu: Vec<(usize, usize)> = Vec::new();
                for &(c, k) in &self.cpu {
                    if off(k, lock.cpu_khz) {
                        match per_cpu.iter_mut().find(|(pc, _)| *pc == c) {
                            Some(slot) => slot.1 += 1,
                            None => per_cpu.push((c, 1)),
                        }
                    }
                }
                let n: usize = per_cpu.iter().map(|&(_, n)| n).sum();
                if n > 0 {
                    let on: Vec<String> = per_cpu
                        .iter()
                        .map(|(c, n)| format!("cpu{c} x{n}"))
                        .collect();
                    v.push(format!(
                        "CPU ran at {:.2}-{:.2} GHz against {:.2} locked ({n} of {} readings \
                         off by more than {:.0}%: {})",
                        lo as f64 / 1e6,
                        hi as f64 / 1e6,
                        lock.cpu_khz as f64 / 1e6,
                        self.cpu.len(),
                        TOL_LOCK * 100.0,
                        on.join(", ")
                    ));
                }
            }
        }
        let want = u64::from(lock.gt_mhz);
        match Self::range(&self.gt) {
            None => v.push("iGPU clock never observed".to_owned()),
            Some((lo, hi)) => {
                let n = self.gt.iter().filter(|&&m| off(m, want)).count();
                if n > 0 {
                    v.push(format!(
                        "iGPU ran at {lo}-{hi} MHz against {want} locked ({n} of {} readings \
                         off by more than {:.0}%)",
                        self.gt.len(),
                        TOL_LOCK * 100.0
                    ));
                }
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: Lock = Lock {
        cpu_khz: 3_000_000,
        gt_mhz: 1400,
    };

    #[test]
    fn no_variables_is_no_lock() {
        assert_eq!(parse_lock(None, None), Ok(None));
    }

    #[test]
    fn both_variables_make_a_lock() {
        assert_eq!(parse_lock(Some("3000000"), Some("1400")), Ok(Some(LOCK)));
    }

    /// A half-specified lock is a mistake in the script, not an unlocked run.
    #[test]
    fn one_variable_alone_is_an_error() {
        assert!(parse_lock(Some("3000000"), None).is_err());
        assert!(parse_lock(None, Some("1400")).is_err());
    }

    #[test]
    fn a_non_numeric_or_zero_value_is_an_error() {
        assert!(parse_lock(Some("3GHz"), Some("1400")).is_err());
        assert!(parse_lock(Some("3000000"), Some("0")).is_err());
    }

    fn settings(cpu_lo: u64, cpu_hi: u64, gt: (u64, u64, u64)) -> Settings {
        Settings {
            cpus: [0, 1, 3]
                .iter()
                .map(|&c| (c, Some(cpu_lo), Some(cpu_hi)))
                .collect(),
            gt: Some((Some(gt.0), Some(gt.1), Some(gt.2))),
        }
    }

    #[test]
    fn a_lock_set_as_requested_is_in_place() {
        let s = settings(3_000_000, 3_000_000, (1400, 1400, 1400));
        assert!(check_in_place(&LOCK, &s).is_empty());
    }

    /// A ceiling is not a lock: with min below max, the governor still
    /// moves the clock.
    #[test]
    fn a_cpu_cap_without_a_floor_is_not_a_lock() {
        let s = settings(400_000, 3_000_000, (1400, 1400, 1400));
        let p = check_in_place(&LOCK, &s);
        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("cpu0"), "{}", p[0]);
    }

    /// Boost counts: i915 can raise the GT to boost over max.
    #[test]
    fn an_igpu_boost_off_the_lock_is_not_a_lock() {
        let s = settings(3_000_000, 3_000_000, (1400, 1400, 2350));
        let p = check_in_place(&LOCK, &s);
        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("boost 2350"), "{}", p[0]);
    }

    #[test]
    fn a_missing_gt_cannot_be_checked() {
        let mut s = settings(3_000_000, 3_000_000, (1400, 1400, 1400));
        s.gt = None;
        assert_eq!(check_in_place(&LOCK, &s).len(), 1);
    }

    fn log(cpu: &[u64], gt: &[u64]) -> ClockLog {
        let mut l = ClockLog::default();
        for (i, &k) in cpu.iter().enumerate() {
            l.record_cpu(i % 2, Some(k));
        }
        for &m in gt {
            l.record_gt(Some(m));
        }
        l
    }

    #[test]
    fn a_lock_that_held_has_no_violations() {
        let l = log(&[2_990_000, 3_000_000, 3_010_000], &[1400, 1400, 1350]);
        assert!(l.violations(&LOCK).is_empty(), "{:?}", l.violations(&LOCK));
    }

    /// The failure the lock exists to catch: the package stepping a unit
    /// down mid-run.
    #[test]
    fn a_unit_stepped_down_mid_run_is_a_violation() {
        let l = log(&[3_000_000, 2_400_000, 3_000_000], &[1400, 1400]);
        let v = l.violations(&LOCK);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("2.40-3.00 GHz"), "{}", v[0]);
        assert!(v[0].contains("1 of 3"), "{}", v[0]);
    }

    #[test]
    fn an_igpu_off_the_lock_is_a_violation() {
        let l = log(&[3_000_000], &[1400, 900]);
        let v = l.violations(&LOCK);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("iGPU"), "{}", v[0]);
    }

    /// Zero is RC6, the GT asleep, not a clock: it is neither counted
    /// against the lock nor as an observation of it.
    #[test]
    fn an_idle_igpu_reading_is_not_a_clock() {
        let mut l = log(&[3_000_000], &[1400]);
        l.record_gt(Some(0));
        assert!(l.violations(&LOCK).is_empty());
        assert_eq!(l.gt_idle, 1);
    }

    fn temp_drm(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("ramvamp-clock-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, "0").unwrap();
    }

    /// The defect this resolution exists for: on a render GT plus media GT
    /// part, the card-level files fan writes out to both GTs. The render GT's
    /// own files must win.
    #[test]
    fn a_multi_gt_card_resolves_to_the_render_gts_own_files() {
        let drm = temp_drm("multi");
        touch(&drm.join("card1/gt_min_freq_mhz"));
        touch(&drm.join("card1/gt/gt0/rps_min_freq_mhz"));
        touch(&drm.join("card1/gt/gt1/rps_min_freq_mhz"));
        touch(&drm.join("card1-eDP-1/status"));
        let gt = find_render_gt(&drm);
        std::fs::remove_dir_all(&drm).unwrap();
        let gt = gt.expect("one render GT");
        assert_eq!(gt.file("min"), drm.join("card1/gt/gt0/rps_min_freq_mhz"));
        assert_eq!(gt.file("act"), drm.join("card1/gt/gt0/rps_act_freq_mhz"));
    }

    /// With one GT there are no per-GT files to prefer, and the card-level
    /// ones are that GT's.
    #[test]
    fn a_single_gt_card_falls_back_to_the_card_level_files() {
        let drm = temp_drm("single");
        touch(&drm.join("card0/gt_min_freq_mhz"));
        let gt = find_render_gt(&drm);
        std::fs::remove_dir_all(&drm).unwrap();
        assert_eq!(
            gt.expect("one GT").file("max"),
            drm.join("card0/gt_max_freq_mhz")
        );
    }

    /// Two i915 cards and no way to tell which one Vulkan picked.
    #[test]
    fn two_cards_are_ambiguous() {
        let drm = temp_drm("two");
        touch(&drm.join("card0/gt/gt0/rps_min_freq_mhz"));
        touch(&drm.join("card1/gt/gt0/rps_min_freq_mhz"));
        let gt = find_render_gt(&drm);
        std::fs::remove_dir_all(&drm).unwrap();
        assert_eq!(gt, None);
    }

    /// Under the GT's lowest active clock is the GT idle or between states:
    /// an active render GT cannot run there, so it is not a slowed GT.
    #[test]
    fn an_igpu_reading_under_its_floor_is_idle_not_slow() {
        let mut l = ClockLog::with_gt_floor(Some(800));
        l.record_cpu(0, Some(3_000_000));
        l.record_gt(Some(1400));
        l.record_gt(Some(100));
        assert!(l.violations(&LOCK).is_empty(), "{:?}", l.violations(&LOCK));
        assert_eq!(l.gt_idle, 1);
        // A step-down to a clock the GT can run at is still caught.
        l.record_gt(Some(900));
        assert_eq!(l.violations(&LOCK).len(), 1);
    }

    /// The off readings are attributed per CPU.
    #[test]
    fn cpu_violations_name_the_cpus() {
        let mut l = ClockLog::default();
        l.record_cpu(0, Some(3_000_000));
        l.record_cpu(3, Some(4_500_000));
        l.record_cpu(3, Some(4_400_000));
        l.record_gt(Some(1400));
        let v = l.violations(&LOCK);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("cpu3 x2"), "{}", v[0]);
        assert!(!v[0].contains("cpu0"), "{}", v[0]);
    }

    #[test]
    fn cpu_lists_parse_as_the_kernel_writes_them() {
        assert_eq!(parse_cpu_list("0,5\n"), Some(vec![0, 5]));
        assert_eq!(parse_cpu_list("1-2"), Some(vec![1, 2]));
        assert_eq!(parse_cpu_list("0-1,8"), Some(vec![0, 1, 8]));
        assert_eq!(parse_cpu_list("7"), Some(vec![7]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a"), None);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let xs: Vec<u64> = (1..=20).collect();
        assert_eq!(ClockLog::pct(&xs, 0.05), 1);
        assert_eq!(ClockLog::pct(&xs, 0.50), 10);
        assert_eq!(ClockLog::pct(&xs, 0.95), 19);
        assert_eq!(ClockLog::pct(&[7], 0.95), 7);
    }

    /// A lock nobody saw hold is not evidence that it held.
    #[test]
    fn a_unit_never_observed_is_a_violation() {
        let mut l = log(&[3_000_000], &[]);
        l.record_gt(Some(0));
        let v = l.violations(&LOCK);
        assert_eq!(v, vec!["iGPU clock never observed".to_owned()]);
    }
}
