//! Run the model commands in their own swap-off cgroup.
//!
//! The common weights are held in anonymous memory (`--common-memory
//! resident`, EXP-031), which nothing can take away from a process whose
//! cgroup has swap turned off; with swap on, the kernel could swap them out
//! when the rest of the machine needs memory instead. So a model command
//! (`generate`, `chat`, `serve`, `logits`) re-launches itself, once, inside a
//! transient systemd user scope with `MemorySwapMax=0`. No root is needed: the
//! user's own systemd manager owns the scope.
//!
//! It is skipped when it cannot help or is not wanted:
//!
//! - the process is already in a cgroup with swap off (`cold_bench.py` runs
//!   every benchmark in one);
//! - `RAMVAMP_NO_SCOPE` is set;
//! - this process is itself the re-launch, whatever the scope turned out to
//!   do, so a scope that could not apply the limit cannot cause a loop;
//! - `systemd-run --user --scope` does not work here (no systemd, no user
//!   session): a no-op probe is tried first, and on failure the command runs
//!   as it is.
//!
//! `--scope` makes `systemd-run` register the cgroup and then `exec` the
//! command in place, so the terminal, signals and exit status are the
//! command's own.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Set on the re-launched process.
const ENV_MARKER: &str = "RAMVAMP_SCOPED";

/// Set by the user to run without the scope.
const ENV_OPT_OUT: &str = "RAMVAMP_NO_SCOPE";

/// What to do about the scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decision {
    /// Swap is already off for this process.
    AlreadyProtected,
    /// `RAMVAMP_NO_SCOPE` is set.
    OptedOut,
    /// This is the re-launch; never re-launch twice.
    AlreadyRelaunched,
    /// Re-launch inside a swap-off scope.
    Relaunch,
}

/// The decision, from the three facts it depends on.
fn decide(relaunched: bool, opted_out: bool, swap_max: Option<&str>) -> Decision {
    if opted_out {
        Decision::OptedOut
    } else if swap_max.map(str::trim) == Some("0") {
        Decision::AlreadyProtected
    } else if relaunched {
        Decision::AlreadyRelaunched
    } else {
        Decision::Relaunch
    }
}

/// This process's cgroup v2 path, from the contents of `/proc/self/cgroup`.
fn cgroup_path(proc_self_cgroup: &str) -> Option<&str> {
    proc_self_cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
}

/// `memory.swap.max` of this process's cgroup, if it can be read.
fn own_swap_max() -> Option<String> {
    let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cgroup_path(&text)?;
    let file = Path::new("/sys/fs/cgroup")
        .join(path.trim_start_matches('/'))
        .join("memory.swap.max");
    std::fs::read_to_string(file).ok()
}

/// The `systemd-run` prefix that starts a command in a swap-off user scope.
fn scope_args() -> [&'static str; 7] {
    [
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        "MemorySwapMax=0",
        "--",
    ]
}

/// Whether `systemd-run --user --scope` works here, by running `true` in one.
fn scope_works() -> bool {
    Command::new("systemd-run")
        .args(scope_args())
        .arg("true")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Re-launch the current command inside a swap-off scope when that helps,
/// and return otherwise.
///
/// Only returns when the command should carry on in this process: on a
/// successful re-launch the process has been replaced. Every failure is
/// logged at debug level and swallowed, because a missing scope costs
/// protection, not correctness.
#[cfg(unix)]
pub fn ensure_swap_off() {
    let decision = decide(
        std::env::var_os(ENV_MARKER).is_some(),
        std::env::var_os(ENV_OPT_OUT).is_some(),
        own_swap_max().as_deref(),
    );
    tracing::debug!(?decision, "swap-off scope");
    if decision != Decision::Relaunch || !scope_works() {
        return;
    }
    let exe: PathBuf = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(err) => {
            tracing::debug!(error = %err, "swap-off scope: cannot find own executable");
            return;
        }
    };
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let err = std::os::unix::process::CommandExt::exec(
        Command::new("systemd-run")
            .args(scope_args())
            .arg(exe)
            .args(args)
            .env(ENV_MARKER, "1"),
    );
    // `exec` only returns on failure.
    tracing::debug!(error = %err, "swap-off scope: re-launch failed; running unscoped");
}

/// No scopes off Unix.
#[cfg(not(unix))]
pub fn ensure_swap_off() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_swap_off_cgroup_is_left_alone() {
        assert_eq!(
            decide(false, false, Some("0\n")),
            Decision::AlreadyProtected
        );
        assert_eq!(decide(true, false, Some("0")), Decision::AlreadyProtected);
    }

    #[test]
    fn the_opt_out_wins_over_everything() {
        assert_eq!(decide(false, true, Some("max")), Decision::OptedOut);
        assert_eq!(decide(false, true, Some("0")), Decision::OptedOut);
        assert_eq!(decide(true, true, None), Decision::OptedOut);
    }

    #[test]
    fn a_relaunch_never_relaunches_again() {
        // The scope ran but its limit did not land (no memory controller
        // delegated, say): carry on rather than loop.
        assert_eq!(
            decide(true, false, Some("max")),
            Decision::AlreadyRelaunched
        );
        assert_eq!(decide(true, false, None), Decision::AlreadyRelaunched);
    }

    #[test]
    fn swap_on_or_unknown_relaunches() {
        assert_eq!(decide(false, false, Some("max")), Decision::Relaunch);
        assert_eq!(decide(false, false, Some("4294967296")), Decision::Relaunch);
        assert_eq!(decide(false, false, None), Decision::Relaunch);
    }

    #[test]
    fn the_cgroup_path_is_the_unified_hierarchy_line() {
        let text = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/x.scope\n";
        assert_eq!(
            cgroup_path(text),
            Some("/user.slice/user-1000.slice/user@1000.service/app.slice/x.scope")
        );
        // Hybrid hierarchies list v1 controllers first.
        let hybrid = "12:memory:/foo\n1:name=systemd:/bar\n0::/baz\n";
        assert_eq!(cgroup_path(hybrid), Some("/baz"));
        assert_eq!(cgroup_path("12:memory:/foo\n"), None);
    }

    #[test]
    fn the_scope_turns_swap_off_and_ends_its_options() {
        let args = scope_args();
        assert!(args.windows(2).any(|w| w == ["-p", "MemorySwapMax=0"]));
        assert_eq!(args.last(), Some(&"--"));
    }
}
