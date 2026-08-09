## What this changes

<!-- What the change does, and why. Link the issue if there is one. -->

## Gate

All four must pass before this is pushed. They are the same four in
`CLAUDE.md`, in the same order.

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets -- -D warnings` with zero warnings
- [ ] `cargo test`
- [ ] `cargo build --release`

## Performance

- [ ] This change cannot affect performance, so no experiment is needed.
- [ ] This change can affect performance, and `docs/experiments.md` has an
      entry for it with a **baseline**, a **result**, and a **verdict**.

If the second box is ticked, the numbers came from a cold run inside a
`memory.max=3G` cgroup with `memory.swap.max=0`, remembering that zram counts
as swap. Warm-cache runs are diagnostics and do not settle anything.

Experiment ID and one-line verdict:

<!-- e.g. EXP-0NN: 1.9 -> 2.1 tok/s decode at ctx 512, cold, in-cgroup. KEEP. -->

## Notes for the reviewer

<!--
Anything that needs context: a design decision, a trade you made, something
you are unsure about, or a measurement that surprised you.

If this touches a vectorized kernel, say which alignment it assumes and where
that assumption is tested. Packed sub-tensor offsets can be as loose as
2-byte aligned.

If it touches expert I/O, say whether the achieved mode in the load banner
changed.
-->
