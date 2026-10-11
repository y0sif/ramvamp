# Routing traces

A recorded routing trace and the per-layer expert strides, so that expert
cache policy work in `scripts/lfu_sim.py` needs no installed model:

```
python3 scripts/lfu_sim.py \
  --trace scripts/traces/qwen3-explain-256.rvtrace \
  --layout scripts/traces/qwen3-30b-a3b-strides.json \
  --pin-step --slots 10,11,12,13 --no-pool-analysis
```

- `qwen3-explain-256.rvtrace`: Qwen3-30B-A3B Q4_K_M, the chat prompt in
  `qwen3-explain-256.prompt.json` (49 prompt tokens), 256 sampled decode
  tokens at seed 2, captured with
  `ramvamp generate --messages-file qwen3-explain-256.prompt.json --max-new 256
  --seed 2 --no-config --trace-experts qwen3-explain-256.rvtrace` on `main`
  plus EXP-028/029 (`c715878`). Format `RVMPTRC1`, documented at the top of
  `scripts/lfu_sim.py`. The live runtime measured 56.7% decode hits on this run
  at 11 slots per layer; the simulator with `--pin-step` reproduces the
  shipped policy's rate on its traces to within 0.01 points (EXP-030).
- `qwen3-30b-a3b-strides.json`: each layer's expert stride from the install's
  `experts/layout.json`, which is all the simulator reads from it. 24 layers
  are 3,059,712 B and 24 are 2,654,208 B.

One trace is a small sample: single-turn and 256 tokens. EXP-030 scored four;
the other three are not committed, to keep the repository small.
