## What this changes

<!-- What the change does, and why. Link the issue if there is one. -->

## Gate

- [ ] Every command in [the gate](https://github.com/y0sif/ramvamp/blob/main/CONTRIBUTING.md#the-gate)
      passes, `./scripts/check-docs.sh` included, with zero clippy warnings.

## Performance

- [ ] This change cannot affect performance.
- [ ] This change can affect performance. Its baseline, result and verdict are
      below, measured by the
      [method in docs/benchmarks.md](https://github.com/y0sif/ramvamp/blob/main/docs/benchmarks.md#method).

<!-- e.g. decode 1.9 -> 2.1 tok/s at ctx 512, cold, in-cgroup, both arms in one session. Keep. -->

## Notes for the reviewer

<!--
Anything that needs context: a design decision, a trade you made, something
you are unsure about, or a measurement that surprised you.

If this touches a vectorized kernel, say which alignment it assumes and where
that assumption is tested.

If it touches expert I/O, say whether the achieved mode in the load banner
changed.
-->
