# Experiment log

Every performance-motivated change gets a numbered entry here before it ships.
This discipline is borrowed from TurboFieldfare's 103-entry experiment record,
which is the reason their claims are credible.

## Rules

1. A microbenchmark starts an experiment; end-to-end speed and output quality
   decide whether it ships.
2. Cold measurements only for published numbers: run inside the benchmark
   cgroup (`memory.max=3G`, `memory.swap.max=0`; zram counts as swap) with a
   dropped page cache.
3. Every entry records its own baseline. Entries use different machine states,
   so numbers from different entries must not be combined into one curve.
4. Changes claiming identical math must produce identical output. Changes that
   reorder floating-point operations must pass tolerance tests against
   reference outputs.
5. Negative results get entries too. They are the cheapest way to stop a bad
   idea from coming back.

## Entry template

```markdown
## EXP-NNN: short title

- Date / commit:
- Hypothesis:
- Method: (workload, machine state, how measured)
- Baseline:
- Result:
- Verdict: KEEP | REVERT | NEUTRAL
- Notes:
```

## Index

(no entries yet)
