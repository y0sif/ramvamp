#!/usr/bin/env python3
"""Offline expert-cache simulator over real Qwen3-30B-A3B routing traces.

Phase 5 wave 0: the per-layer expert cache does not exist yet, and its hit
rate is the single number that sets the runtime's performance ceiling
(every miss is a `stride`-byte read from NVMe). This replays captured
routing traces through candidate cache configurations and reports what the
real routing behaviour actually permits, so the slot budget and the
eviction policy are chosen from measurement rather than from upstream
folklore.

Inputs:

  * one or more binary traces from `ramvamp generate --trace-experts`
  * `models/qwen3.rvmp/experts/layout.json`, for the real per-layer expert
    stride (Qwen3-30B-A3B is not uniform: 24 layers are 3,059,712 B and 24
    are 2,654,208 B, interleaved), which turns hit rates into bytes and
    slot counts into MiB

Model of the cache under test (`docs/architecture.md`, "Expert streaming
and cache"): one independent slot array per layer, `--slots` entries each,
fetch on miss, evict by policy. Prefill bypasses the cache, so only decode
records are simulated by default. Experts already fetched for the token
being decoded are pinned for that step: a later miss in the same layer and
token cannot evict them (the runtime is holding those slabs).

Trace format `RVMPTRC1`, written by `TraceWriter` in `crates/cli/src/main.rs`
(which documents the same layout); every integer little-endian:

    header, 28 bytes
      magic      [u8; 8]  b"RVMPTRC1"
      version    u32      1
      n_layers   u32      layers per record
      n_experts  u32      routed experts per layer (the id space)
      top_k      u32      routed experts per layer per token
      n_records  u32      complete records
    record, 8 + n_layers * top_k * 4 bytes, repeated n_records times
      phase      u8       0 = prefill, 1 = decode
      _pad       [u8; 3]  zero
      position   u32      sequence position of the token
      experts    [u32; n_layers * top_k]
                          layer-major, layer 0 first; within a layer the
                          top_k ids in routed order, i.e. descending
                          router probability

Python stdlib only, like the repo's other scripts.

Example:

    uv run scripts/lfu_sim.py \
        --layout models/qwen3.rvmp/experts/layout.json \
        --trace traces/*.rvtrace \
        --out traces/lfu_results.json
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import struct
import sys

# Measured sequential read ceiling of the expert blobs on the dev NVMe.
DEFAULT_BANDWIDTH_GBPS = 1.59
# Slot budgets swept by default (slots per layer). 8/16/24/32 are the slot
# counts TurboFieldfare ships and published hit rates for.
DEFAULT_SLOTS = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48)
# Eviction policies compared by default.
DEFAULT_POLICIES = ("lfu", "lfu-ghost", "lru", "lfu-aged", "lfu-window", "opt")
# Accesses per layer between halvings of the LFU counters in `lfu-aged`.
DEFAULT_AGE_PERIOD = 256
# Window of `lfu-window`, in decode tokens (TurboFieldfare tested 64).
DEFAULT_WINDOW_TOKENS = 64

# Everything else that has to fit next to the slot pool in the published
# 3 GB cgroup (`docs/architecture.md`, "Memory budget").
COMMON_WEIGHTS_MIB = 1023.34
KV_CACHE_MIB = 384.0
CGROUP_MIB = 3072.0

# TurboFieldfare's published figures on the same model shape (128 experts,
# top-8, per-layer slot arrays), for a like-for-like comparison. Ship
# default 16 slots/layer; allowed values 8/16/24/32.
TF_HIT_AT_16 = 0.666
TF_DELTA_16_TO_24 = (0.126, 0.149)
TF_DELTA_16_TO_32 = (0.209, 0.237)
# Belady headroom TurboFieldfare measured over their own LFU, in points.
TF_OPT_HEADROOM = (0.08, 0.11)

# Reuse-distance buckets for eviction misses, in decode tokens.
REUSE_BUCKETS = (1, 4, 16, 64)

TRACE_MAGIC = b"RVMPTRC1"
TRACE_VERSION = 1
HEADER_BYTES = 28
PHASE_PREFILL = 0
PHASE_DECODE = 1


def fail(message: str) -> None:
    """Abort with a message on stderr."""
    print(f"lfu_sim: {message}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


class Trace:
    """One captured generation: header facts plus per-token routing."""

    def __init__(
        self,
        name: str,
        n_layers: int,
        n_experts: int,
        top_k: int,
        phases: array.array,
        positions: array.array,
        experts: array.array,
    ) -> None:
        self.name = name
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        # Parallel arrays, one entry per record.
        self.phases = phases
        self.positions = positions
        # Flat [record][layer][k] expert ids.
        self.experts = experts

    @property
    def n_records(self) -> int:
        return len(self.phases)

    @property
    def ids_per_record(self) -> int:
        return self.n_layers * self.top_k

    def routed(self, record: int, layer: int) -> array.array:
        """The `top_k` ids routed at `record` in `layer`, routed order."""
        base = record * self.ids_per_record + layer * self.top_k
        return self.experts[base : base + self.top_k]

    def records_of(self, phase: int) -> list[int]:
        """Record indices with the given phase, in stream order."""
        return [i for i, p in enumerate(self.phases) if p == phase]


def read_trace(path: str) -> Trace:
    """Parse one `RVMPTRC1` file (format documented in the module docstring)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < HEADER_BYTES:
        fail(f"{path}: {len(blob)} bytes, shorter than the {HEADER_BYTES}-byte header")
    if blob[:8] != TRACE_MAGIC:
        fail(f"{path}: bad magic {blob[:8]!r}, expected {TRACE_MAGIC!r}")
    version, n_layers, n_experts, top_k, n_records = struct.unpack_from("<5I", blob, 8)
    if version != TRACE_VERSION:
        fail(f"{path}: trace version {version}, this reader speaks {TRACE_VERSION}")
    if n_layers == 0 or top_k == 0:
        fail(f"{path}: degenerate header (n_layers={n_layers}, top_k={top_k})")

    ids_per_record = n_layers * top_k
    record_bytes = 8 + ids_per_record * 4
    want = HEADER_BYTES + n_records * record_bytes
    if len(blob) < want:
        fail(f"{path}: header claims {n_records} records ({want} B), file is {len(blob)} B")

    phases = array.array("B", bytes(n_records))
    positions = array.array("I", bytes(4 * n_records))
    experts = array.array("I", bytes(4 * ids_per_record * n_records))
    for r in range(n_records):
        base = HEADER_BYTES + r * record_bytes
        phases[r] = blob[base]
        positions[r] = struct.unpack_from("<I", blob, base + 4)[0]
        chunk = array.array("I")
        chunk.frombytes(blob[base + 8 : base + record_bytes])
        if sys.byteorder != "little":
            chunk.byteswap()
        experts[r * ids_per_record : (r + 1) * ids_per_record] = chunk

    bad = [e for e in experts if e >= n_experts]
    if bad:
        fail(f"{path}: expert id {bad[0]} outside the {n_experts}-expert id space")
    return Trace(os.path.basename(path), n_layers, n_experts, top_k, phases, positions, experts)


def read_strides(path: str) -> list[int]:
    """Per-layer expert stride in bytes, from an install's layout.json."""
    with open(path) as fh:
        layout = json.load(fh)
    layers = layout.get("layers")
    if not layers:
        fail(f"{path}: no layers in layout")
    return [int(layer["stride"]) for layer in layers]


# --------------------------------------------------------------------------
# Eviction policies
# --------------------------------------------------------------------------
#
# Each cache is one layer's slot array. `access(expert, clock, pinned)`
# returns `(hit, victim)`: `hit` is whether the slab was already resident,
# and `victim` is the expert this access evicted, if any (the caller needs
# it to classify later misses and to measure reuse distance). On a miss the
# expert is inserted, evicting the policy's victim from the entries not
# pinned by the token in flight.

Access = tuple[bool, "int | None"]


class LfuCache:
    """LFU with an LRU tie-break, counters scoped to the SLOT.

    A count exists only while its expert is resident: evicting an expert
    throws its history away and a re-fetched slab starts at 1. This is what
    a naive reading of `docs/architecture.md` ("LFU eviction with recency
    tie-break") produces, and it is deliberately NOT what the upstream
    implementation does - see `GhostLfuCache`.
    """

    label = "lfu"

    def __init__(self, slots: int, **_: object) -> None:
        self.slots = slots
        self.count: dict[int, int] = {}
        self.used: dict[int, int] = {}

    def access(self, expert: int, clock: int, pinned: set[int]) -> Access:
        if expert in self.count:
            self.count[expert] += 1
            self.used[expert] = clock
            return True, None
        return False, self._insert(expert, clock, pinned)

    def _victim(self, pinned: set[int]) -> int | None:
        best, best_key = None, None
        for e in self.count:
            if e in pinned:
                continue
            key = (self.count[e], self.used[e])
            if best_key is None or key < best_key:
                best, best_key = e, key
        return best

    def _admit(self, expert: int, clock: int) -> None:
        self.count[expert] = 1
        self.used[expert] = clock

    def _insert(self, expert: int, clock: int, pinned: set[int]) -> int | None:
        victim = None
        if len(self.count) >= self.slots:
            victim = self._victim(pinned)
            if victim is None:
                # Every slot is held by this token: the slab is read and
                # dropped, exactly as the runtime would have to.
                return None
            del self.count[victim]
            del self.used[victim]
        self._admit(expert, clock)
        return victim


class GhostLfuCache(LfuCache):
    """LFU with full ghost history: counters indexed by expert, never reset.

    This is what TurboFieldfare actually ships (`expertUseCount` is sized
    to the 128 experts of a layer, not to the slot count), and it is a
    materially different policy from `LfuCache`: a count survives eviction,
    so an expert that was hot once is hard to evict ever again. Counters
    are monotonic and are bumped on every routing decision, hit or miss.
    """

    label = "lfu-ghost"

    def __init__(self, slots: int, n_experts: int = 0, **_: object) -> None:
        super().__init__(slots)
        # Persistent, never decayed, never cleared on eviction.
        self.ghost: dict[int, int] = {}
        self.n_experts = n_experts
        # Resident set; `self.count` is unused, `self.used` keeps recency.
        self.resident: set[int] = set()

    def access(self, expert: int, clock: int, pinned: set[int]) -> Access:
        self.ghost[expert] = self.ghost.get(expert, 0) + 1
        if expert in self.resident:
            self.used[expert] = clock
            return True, None
        victim = None
        if len(self.resident) >= self.slots:
            best, best_key = None, None
            for e in self.resident:
                if e in pinned:
                    continue
                key = (self.ghost.get(e, 0), self.used[e])
                if best_key is None or key < best_key:
                    best, best_key = e, key
            if best is None:
                return False, None
            victim = best
            self.resident.discard(victim)
            del self.used[victim]
        self.resident.add(expert)
        self.used[expert] = clock
        return False, victim


class LruCache(LfuCache):
    """Plain LRU: victim is the least recently used entry."""

    label = "lru"

    def _victim(self, pinned: set[int]) -> int | None:
        best, best_key = None, None
        for e in self.used:
            if e in pinned:
                continue
            if best_key is None or self.used[e] < best_key:
                best, best_key = e, self.used[e]
        return best


class AgedLfuCache(LfuCache):
    """LFU whose counters halve every `age_period` accesses to this layer.

    Classic LFU-with-aging: without decay an expert that was hot early
    keeps a count no newcomer can beat, so the cache freezes around the
    routing of the first few tokens. Halving bounds how much history one
    entry can bank.
    """

    label = "lfu-aged"

    def __init__(self, slots: int, age_period: int = DEFAULT_AGE_PERIOD, **_: object) -> None:
        super().__init__(slots)
        self.age_period = max(1, age_period)
        self.since_decay = 0

    def access(self, expert: int, clock: int, pinned: set[int]) -> Access:
        out = super().access(expert, clock, pinned)
        self.since_decay += 1
        if self.since_decay >= self.age_period:
            self.since_decay = 0
            for e in self.count:
                self.count[e] = max(1, self.count[e] // 2)
        return out


class WindowLfuCache(LfuCache):
    """LFU over a sliding window of the most recent accesses.

    The TinyLFU-flavoured variant: frequency is counted only over the last
    `window_tokens` decode tokens (`window_tokens * top_k` accesses in this
    layer), so old popularity expires outright instead of decaying.
    Upstream tested a 64-token window, saw fewer simulated misses, and
    still rejected it on real decode - worth reproducing here.
    """

    label = "lfu-window"

    def __init__(
        self,
        slots: int,
        window_tokens: int = DEFAULT_WINDOW_TOKENS,
        top_k: int = 8,
        **_: object,
    ) -> None:
        super().__init__(slots)
        self.window = max(1, window_tokens * top_k)
        self.recent: list[int] = []
        self.head = 0
        self.freq: dict[int, int] = {}

    def _push(self, expert: int) -> None:
        self.recent.append(expert)
        self.freq[expert] = self.freq.get(expert, 0) + 1
        if len(self.recent) - self.head > self.window:
            old = self.recent[self.head]
            self.head += 1
            self.freq[old] -= 1
            if self.freq[old] <= 0:
                del self.freq[old]
            if self.head > self.window:
                self.recent = self.recent[self.head :]
                self.head = 0

    def access(self, expert: int, clock: int, pinned: set[int]) -> Access:
        self._push(expert)
        if expert in self.count:
            self.count[expert] += 1
            self.used[expert] = clock
            return True, None
        return False, self._insert(expert, clock, pinned)

    def _victim(self, pinned: set[int]) -> int | None:
        best, best_key = None, None
        for e in self.count:
            if e in pinned:
                continue
            key = (self.freq.get(e, 0), self.used[e])
            if best_key is None or key < best_key:
                best, best_key = e, key
        return best


class OptCache:
    """Belady's optimal replacement: evict whatever is needed farthest away.

    Not implementable online (it reads the future), but it is the ceiling
    every real policy is measured against: the gap between `lfu` and `opt`
    is all the hit rate a smarter policy could still buy at that slot count.
    """

    label = "opt"

    def __init__(self, slots: int, future: list[list[int]] | None = None, **_: object) -> None:
        self.slots = slots
        self.resident: set[int] = set()
        # future[e] = ascending access clocks for expert e in this layer.
        self.future = future or []
        self.cursor = [0] * len(self.future)

    def _next_use(self, expert: int, clock: int) -> float:
        uses = self.future[expert]
        i = self.cursor[expert]
        while i < len(uses) and uses[i] <= clock:
            i += 1
        return uses[i] if i < len(uses) else math.inf

    def access(self, expert: int, clock: int, pinned: set[int]) -> Access:
        # Consume this access from the expert's future list.
        uses = self.future[expert]
        i = self.cursor[expert]
        while i < len(uses) and uses[i] < clock:
            i += 1
        if i < len(uses) and uses[i] == clock:
            i += 1
        self.cursor[expert] = i

        if expert in self.resident:
            return True, None
        victim = None
        if len(self.resident) >= self.slots:
            farthest = -1.0
            for e in self.resident:
                if e in pinned:
                    continue
                nxt = self._next_use(e, clock)
                if nxt > farthest:
                    victim, farthest = e, nxt
            if victim is None:
                return False, None
            self.resident.discard(victim)
        self.resident.add(expert)
        return False, victim


POLICIES = {
    LfuCache.label: LfuCache,
    GhostLfuCache.label: GhostLfuCache,
    LruCache.label: LruCache,
    AgedLfuCache.label: AgedLfuCache,
    WindowLfuCache.label: WindowLfuCache,
    OptCache.label: OptCache,
}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def opt_future(trace: Trace, records: list[int], n_experts: int) -> list[list[list[int]]]:
    """Per-layer, per-expert ascending access clocks, for `OptCache`."""
    future = [[[] for _ in range(n_experts)] for _ in range(trace.n_layers)]
    clock = 0
    for r in records:
        for layer in range(trace.n_layers):
            for e in trace.routed(r, layer):
                future[layer][e].append(clock)
                clock += 1
    return future


def simulate(
    traces: list[Trace],
    strides: list[int],
    slots: int,
    policy: str,
    age_period: int,
    warm_from_prefill: bool,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> dict:
    """Replay every trace's decode records through one cache configuration.

    Each trace starts from an empty cache (a generation is one session).
    With `warm_from_prefill` the prefill records are replayed first to warm
    the cache; their own hits and misses are not scored, because the design
    bypasses the cache during prefill.

    Misses are split into cold (this layer has never fetched that expert in
    this generation, so no cache size would have helped) and eviction (it
    was resident and got thrown out, so a bigger cache would have helped).
    Every eviction miss also contributes its reuse distance, in decode
    tokens between the eviction and the re-request.
    """
    cls = POLICIES[policy]
    n_layers = traces[0].n_layers
    n_experts = traces[0].n_experts
    top_k = traces[0].top_k
    hits = [0] * n_layers
    misses = [0] * n_layers
    cold_misses = [0] * n_layers
    evict_misses = [0] * n_layers
    reuse_hist = [0] * (len(REUSE_BUCKETS) + 1)
    reuse_layer_hist = [[0] * (len(REUSE_BUCKETS) + 1) for _ in range(n_layers)]
    miss_bytes = 0
    tokens = 0
    # Cumulative decode-token hit counts, for the cold-start curve.
    per_token_hits: list[int] = []
    per_token_accesses: list[int] = []

    for trace in traces:
        decode = trace.records_of(PHASE_DECODE)
        prefill = trace.records_of(PHASE_PREFILL) if warm_from_prefill else []
        if policy == OptCache.label:
            future = opt_future(trace, prefill + decode, n_experts)
            caches = [cls(slots, future=future[layer]) for layer in range(n_layers)]
        else:
            caches = [
                cls(
                    slots,
                    age_period=age_period,
                    n_experts=n_experts,
                    window_tokens=window_tokens,
                    top_k=top_k,
                )
                for _ in range(n_layers)
            ]
        # Per layer: experts ever fetched, and the token each eviction
        # happened at (so a later miss can report its reuse distance).
        fetched: list[set[int]] = [set() for _ in range(n_layers)]
        evicted_at: list[dict[int, int]] = [{} for _ in range(n_layers)]

        clock = 0
        for r in prefill:
            for layer in range(n_layers):
                pinned: set[int] = set()
                for e in trace.routed(r, layer):
                    _, victim = caches[layer].access(e, clock, pinned)
                    fetched[layer].add(e)
                    if victim is not None:
                        evicted_at[layer][victim] = 0
                    pinned.add(e)
                    clock += 1

        for t, r in enumerate(decode):
            token_hits = 0
            token_accesses = 0
            for layer in range(n_layers):
                cache = caches[layer]
                pinned = set()
                for e in trace.routed(r, layer):
                    hit, victim = cache.access(e, clock, pinned)
                    if hit:
                        hits[layer] += 1
                        token_hits += 1
                    else:
                        misses[layer] += 1
                        miss_bytes += strides[layer]
                        if e in fetched[layer]:
                            evict_misses[layer] += 1
                            distance = t - evicted_at[layer].get(e, 0)
                            bucket = len(REUSE_BUCKETS)
                            for i, edge in enumerate(REUSE_BUCKETS):
                                if distance <= edge:
                                    bucket = i
                                    break
                            reuse_hist[bucket] += 1
                            reuse_layer_hist[layer][bucket] += 1
                        else:
                            cold_misses[layer] += 1
                            fetched[layer].add(e)
                    if victim is not None:
                        evicted_at[layer][victim] = t
                    pinned.add(e)
                    clock += 1
                    token_accesses += 1
            tokens += 1
            if t < len(per_token_hits):
                per_token_hits[t] += token_hits
                per_token_accesses[t] += token_accesses
            else:
                per_token_hits.append(token_hits)
                per_token_accesses.append(token_accesses)

    total_hits = sum(hits)
    total = total_hits + sum(misses)
    layer_rates = [
        hits[l] / (hits[l] + misses[l]) if hits[l] + misses[l] else 0.0 for l in range(n_layers)
    ]
    return {
        "policy": policy,
        "slots": slots,
        "warm_from_prefill": warm_from_prefill,
        "decode_tokens": tokens,
        "accesses": total,
        "hits": total_hits,
        "hit_rate": total_hits / total if total else 0.0,
        "layer_hit_rate": layer_rates,
        "cold_misses": sum(cold_misses),
        "eviction_misses": sum(evict_misses),
        "layer_cold_misses": cold_misses,
        "layer_eviction_misses": evict_misses,
        "reuse_histogram": reuse_hist,
        "layer_reuse_histogram": reuse_layer_hist,
        "bytes_per_token": miss_bytes / tokens if tokens else 0.0,
        "memory_bytes": slots * sum(strides),
        "per_token_hits": per_token_hits,
        "per_token_accesses": per_token_accesses,
    }


# --------------------------------------------------------------------------
# Routing statistics (why the cache behaves the way it does)
# --------------------------------------------------------------------------


def layer_curves(
    traces: list[Trace], slots_max: int, policy: str, age_period: int
) -> tuple[list[list[int]], int]:
    """Per-layer hit counts for every slot count in `1..=slots_max`.

    Feeds the global-pool question: `docs/architecture.md` lists "global
    slot pool vs per-layer (hot layers steal slots)" as an open decision,
    and it can only be answered if each layer's own hit curve is known.
    Returns `(hits[layer][slots - 1], accesses_per_layer)`.
    """
    cls = POLICIES[policy]
    n_layers = traces[0].n_layers
    n_experts = traces[0].n_experts
    curves = [[0] * slots_max for _ in range(n_layers)]
    accesses = 0
    for trace in traces:
        decode = trace.records_of(PHASE_DECODE)
        accesses += len(decode) * trace.top_k * n_layers
        for layer in range(n_layers):
            routed = [trace.routed(r, layer) for r in decode]
            for slots in range(1, slots_max + 1):
                if policy == OptCache.label:
                    future = [[] for _ in range(n_experts)]
                    clock = 0
                    for ids in routed:
                        for e in ids:
                            future[e].append(clock)
                            clock += 1
                    cache = cls(slots, future=future)
                else:
                    cache = cls(
                        slots,
                        age_period=age_period,
                        n_experts=n_experts,
                        top_k=trace.top_k,
                    )
                clock = 0
                hits = 0
                for ids in routed:
                    pinned: set[int] = set()
                    for e in ids:
                        if cache.access(e, clock, pinned)[0]:
                            hits += 1
                        pinned.add(e)
                        clock += 1
                curves[layer][slots - 1] += hits
    return curves, accesses


def concave_gains(curve: list[int]) -> list[float]:
    """Per-slot marginal hits, forced non-increasing (pool adjacent violators).

    A greedy allocator only reaches the optimum when each layer's marginal
    return is diminishing. Measured LFU curves are close to that but not
    exactly, so the allocator ranks on this concave majorant and the
    resulting allocation is then scored against the true curve.
    """
    gains = [curve[0]] + [curve[i] - curve[i - 1] for i in range(1, len(curve))]
    blocks: list[list[float]] = []
    for g in gains:
        blocks.append([float(g), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    out: list[float] = []
    for total, count in blocks:
        out.extend([total / count] * int(count))
    return out


def greedy_pool(curves: list[list[int]], budget: int) -> tuple[list[int], int]:
    """Best static per-layer split of `budget` total slots, greedily.

    Each layer starts at one slot; the next slot goes wherever the concave
    majorant says it buys the most hits. The result bounds what a shared
    pool with a static split could deliver over the uniform allocation the
    design assumes ("global slot pool vs per-layer" in the architecture's
    open decisions).
    """
    n_layers = len(curves)
    slots_max = len(curves[0])
    if budget < n_layers:
        return [0] * n_layers, 0
    ranked = [concave_gains(c) for c in curves]
    alloc = [1] * n_layers
    for _ in range(budget - n_layers):
        best_layer, best_gain = None, 0.0
        for layer in range(n_layers):
            s = alloc[layer]
            if s >= slots_max:
                continue
            if ranked[layer][s] > best_gain:
                best_layer, best_gain = layer, ranked[layer][s]
        if best_layer is None:
            break
        alloc[best_layer] += 1
    return alloc, sum(curves[l][alloc[l] - 1] for l in range(n_layers))


def routing_stats(traces: list[Trace], strides: list[int], warmup: int) -> dict:
    """Skew, reuse, and infinite-cache statistics over the decode records."""
    n_layers = traces[0].n_layers
    n_experts = traces[0].n_experts
    top_k = traces[0].top_k

    counts = [[0] * n_experts for _ in range(n_layers)]
    seen_before = [0] * n_layers
    accesses = [0] * n_layers
    consecutive_hits = [0] * n_layers
    consecutive_total = [0] * n_layers
    inf_after_warmup_hits = 0
    inf_after_warmup_total = 0
    decode_tokens = 0

    for trace in traces:
        decode = trace.records_of(PHASE_DECODE)
        decode_tokens += len(decode)
        seen: list[set[int]] = [set() for _ in range(n_layers)]
        prev: list[set[int]] | None = None
        for t, r in enumerate(decode):
            cur: list[set[int]] = []
            for layer in range(n_layers):
                routed = trace.routed(r, layer)
                cur.append(set(routed))
                for e in routed:
                    counts[layer][e] += 1
                    accesses[layer] += 1
                    if e in seen[layer]:
                        seen_before[layer] += 1
                        if t >= warmup:
                            inf_after_warmup_hits += 1
                    else:
                        seen[layer].add(e)
                    if t >= warmup:
                        inf_after_warmup_total += 1
                if prev is not None:
                    consecutive_hits[layer] += len(cur[layer] & prev[layer])
                    consecutive_total[layer] += len(cur[layer])
            prev = cur

    per_layer = []
    for layer in range(n_layers):
        c = sorted((n for n in counts[layer] if n), reverse=True)
        total = accesses[layer] or 1
        entropy = -sum((n / total) * math.log2(n / total) for n in c)
        per_layer.append(
            {
                "layer": layer,
                "stride": strides[layer],
                "distinct_experts": len(c),
                "entropy_bits": entropy,
                "entropy_normalized": entropy / math.log2(n_experts),
                "top8_coverage": sum(c[:8]) / total,
                "top16_coverage": sum(c[:16]) / total,
                "top32_coverage": sum(c[:32]) / total,
                "infinite_cache_hit_rate": seen_before[layer] / total,
                "consecutive_reuse": (
                    consecutive_hits[layer] / consecutive_total[layer]
                    if consecutive_total[layer]
                    else 0.0
                ),
            }
        )

    total_acc = sum(accesses) or 1
    return {
        "decode_tokens": decode_tokens,
        "top_k": top_k,
        "n_experts": n_experts,
        "uniform_entropy_bits": math.log2(n_experts),
        "mean_entropy_bits": sum(p["entropy_bits"] for p in per_layer) / n_layers,
        "mean_top8_coverage": sum(p["top8_coverage"] for p in per_layer) / n_layers,
        "mean_top16_coverage": sum(p["top16_coverage"] for p in per_layer) / n_layers,
        "mean_top32_coverage": sum(p["top32_coverage"] for p in per_layer) / n_layers,
        "mean_distinct_experts": sum(p["distinct_experts"] for p in per_layer) / n_layers,
        "consecutive_reuse": sum(consecutive_hits) / (sum(consecutive_total) or 1),
        "infinite_cache_hit_rate": sum(seen_before) / total_acc,
        "infinite_cache_hit_rate_after_warmup": (
            inf_after_warmup_hits / inf_after_warmup_total if inf_after_warmup_total else 0.0
        ),
        "warmup_tokens": warmup,
        "per_layer": per_layer,
    }


def cold_start(result: dict, window: int, tolerance: float, limit: int) -> dict:
    """When the running hit rate settles inside `tolerance` of steady state.

    Steady state is the hit rate over the last half of the decode tokens;
    the answer is the first windowed hit rate that reaches it and stays.
    The curve is truncated to `limit` (the shortest trace's decode length)
    so every window averages the same set of traces - otherwise the tail
    would silently switch to whichever generations ran longest.
    """
    hits = result["per_token_hits"][:limit]
    accesses = result["per_token_accesses"][:limit]
    n = len(hits)
    if n == 0:
        return {"windows": [], "steady_state": 0.0, "tokens_to_steady": None, "limit": limit}
    half = n // 2
    steady_h = sum(hits[half:])
    steady_a = sum(accesses[half:]) or 1
    steady = steady_h / steady_a

    windows = []
    for start in range(0, n, window):
        stop = min(start + window, n)
        h = sum(hits[start:stop])
        a = sum(accesses[start:stop]) or 1
        windows.append({"from_token": start, "to_token": stop, "hit_rate": h / a})

    tokens_to_steady = None
    for i, w in enumerate(windows):
        if all(abs(v["hit_rate"] - steady) <= tolerance for v in windows[i:]):
            tokens_to_steady = w["from_token"]
            break
    cumulative = []
    run_h = run_a = 0
    for t in range(n):
        run_h += hits[t]
        run_a += accesses[t]
        if (t + 1) % window == 0 or t == n - 1:
            cumulative.append({"tokens": t + 1, "hit_rate": run_h / (run_a or 1)})
    return {
        "window": window,
        "tolerance": tolerance,
        "limit": n,
        "steady_state": steady,
        "tokens_to_steady": tokens_to_steady,
        "windows": windows,
        "cumulative": cumulative,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def mib(byte_count: float) -> float:
    return byte_count / (1024 * 1024)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    i = q * (len(sorted_values) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (i - lo)


def io_seconds(bytes_per_token: float, bandwidth_gbps: float) -> float:
    return bytes_per_token / (bandwidth_gbps * 1e9)


def print_sweep(results: list[dict], bandwidth: float, slots_list: list[int]) -> None:
    print()
    print("=" * 96)
    print("SLOT SWEEP  (hit rate, I/O per decode token, and the memory it costs)")
    print("=" * 96)
    header = (
        f"{'policy':<11} {'slots':>5} {'pool MiB':>9} {'hit %':>7} "
        f"{'layer hit % p10/p50/p90':>24} "
        f"{'MB/tok':>8} {'io ms/tok':>10} {'io-only tok/s':>13}"
    )
    print(header)
    print("-" * len(header))
    for policy in dict.fromkeys(r["policy"] for r in results):
        for slots in slots_list:
            r = next(
                (x for x in results if x["policy"] == policy and x["slots"] == slots),
                None,
            )
            if r is None:
                continue
            lr = sorted(r["layer_hit_rate"])
            secs = io_seconds(r["bytes_per_token"], bandwidth)
            print(
                f"{policy:<11} {slots:>5} {mib(r['memory_bytes']):>9.0f} "
                f"{100 * r['hit_rate']:>7.2f} "
                f"{100 * quantile(lr, 0.1):>7.1f}/{100 * quantile(lr, 0.5):>6.1f}/"
                f"{100 * quantile(lr, 0.9):>6.1f}  "
                f"{r['bytes_per_token'] / 1e6:>8.1f} {1000 * secs:>10.1f} "
                f"{(1 / secs if secs else float('inf')):>13.2f}"
            )
        print()


def print_budget(results: list[dict], slots_list: list[int], policy: str) -> list[dict]:
    """Each slot count as a whole-process memory budget against the cgroup.

    The published numbers come from a `memory.max=3G` cgroup, so the only
    slot counts that matter are the ones whose pool still leaves room for
    the mmap'd common core and the KV cache.
    """
    print("=" * 96)
    print(f"MEMORY BUDGET vs THE 3G CGROUP  (policy {policy})")
    print("=" * 96)
    print(
        f"common core {COMMON_WEIGHTS_MIB:.0f} MiB (mmap) + KV cache "
        f"{KV_CACHE_MIB:.0f} MiB @ 4K + slot pool, against {CGROUP_MIB:.0f} MiB"
    )
    header = (
        f"{'slots':>5} {'pool MiB':>9} {'total MiB':>10} {'headroom MiB':>13} "
        f"{'fits 3G':>8} {'hit %':>7}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for slots in slots_list:
        r = next(
            (x for x in results if x["policy"] == policy and x["slots"] == slots),
            None,
        )
        if r is None:
            continue
        pool = mib(r["memory_bytes"])
        total = pool + COMMON_WEIGHTS_MIB + KV_CACHE_MIB
        fits = total <= CGROUP_MIB
        rows.append(
            {
                "slots": slots,
                "pool_mib": pool,
                "total_mib": total,
                "headroom_mib": CGROUP_MIB - total,
                "fits_3g": fits,
                "hit_rate": r["hit_rate"],
            }
        )
        print(
            f"{slots:>5} {pool:>9.0f} {total:>10.0f} {CGROUP_MIB - total:>13.0f} "
            f"{('yes' if fits else 'NO'):>8} {100 * r['hit_rate']:>7.2f}"
        )
    print()
    return rows


def print_reference(results: list[dict], policies: list[str]) -> dict:
    """Our curve at TurboFieldfare's shipped slot counts, against theirs."""
    print("=" * 96)
    print("UPSTREAM REFERENCE POINTS  (TurboFieldfare: 128 experts, top-8, per-layer slots)")
    print("=" * 96)

    def at(policy: str, slots: int) -> float | None:
        r = next(
            (x for x in results if x["policy"] == policy and x["slots"] == slots),
            None,
        )
        return None if r is None else r["hit_rate"]

    header = f"{'slots':>5} " + " ".join(f"{p:>12}" for p in policies) + f"{'  upstream':>12}"
    print(header)
    print("-" * len(header))
    for slots in (8, 10, 16, 24, 32):
        cells = []
        for policy in policies:
            v = at(policy, slots)
            cells.append(f"{'-':>12}" if v is None else f"{100 * v:>11.2f}%")
        upstream = f"{100 * TF_HIT_AT_16:>11.1f}%" if slots == 16 else f"{'-':>12}"
        tag = "  <- their ship default" if slots == 16 else ""
        tag = "  <- our design doc" if slots == 10 else tag
        print(f"{slots:>5} " + " ".join(cells) + upstream + tag)

    out: dict = {"upstream_hit_at_16": TF_HIT_AT_16, "policies": {}}
    print()
    print(f"{'policy':<12} {'16->24 points':>15} {'upstream':>18} {'16->32 points':>15} {'upstream':>18}")
    print("-" * 82)
    for policy in policies:
        h16, h24, h32 = at(policy, 16), at(policy, 24), at(policy, 32)
        if h16 is None or h24 is None or h32 is None:
            continue
        d24, d32 = 100 * (h24 - h16), 100 * (h32 - h16)
        out["policies"][policy] = {
            "hit_at_8": at(policy, 8),
            "hit_at_10": at(policy, 10),
            "hit_at_16": h16,
            "hit_at_24": h24,
            "hit_at_32": h32,
            "delta_16_to_24": h24 - h16,
            "delta_16_to_32": h32 - h16,
        }
        print(
            f"{policy:<12} {d24:>15.2f} "
            f"{f'{100 * TF_DELTA_16_TO_24[0]:.1f}-{100 * TF_DELTA_16_TO_24[1]:.1f}':>18} "
            f"{d32:>15.2f} "
            f"{f'{100 * TF_DELTA_16_TO_32[0]:.1f}-{100 * TF_DELTA_16_TO_32[1]:.1f}':>18}"
        )
    print()
    return out


def print_misses(results: list[dict], slots_list: list[int], policy: str) -> None:
    """Cold vs eviction misses, and the reuse distance of the evicted ones."""
    print("=" * 96)
    print(f"MISS CLASSIFICATION  (policy {policy})")
    print("=" * 96)
    print("cold = never fetched in this generation (no cache size helps);")
    print("eviction = was resident and thrown out (a bigger cache would have kept it).")
    labels = [f"<={b}" for b in REUSE_BUCKETS] + [f">{REUSE_BUCKETS[-1]}"]
    print("reuse distance = decode tokens between an expert's eviction and its re-request.")
    header = (
        f"{'slots':>5} {'misses':>9} {'cold %':>7} {'evict %':>8} | "
        + " ".join(f"{l:>7}" for l in labels)
    )
    print(header)
    print("-" * len(header))
    for slots in slots_list:
        r = next(
            (x for x in results if x["policy"] == policy and x["slots"] == slots),
            None,
        )
        if r is None:
            continue
        total = r["accesses"] - r["hits"]
        cold, evict = r["cold_misses"], r["eviction_misses"]
        hist = r["reuse_histogram"]
        denom = sum(hist) or 1
        cells = " ".join(f"{100 * h / denom:>6.1f}%" for h in hist)
        print(
            f"{slots:>5} {total:>9} {100 * cold / (total or 1):>6.1f}% "
            f"{100 * evict / (total or 1):>7.1f}% | " + cells
        )
    print()


def print_policy_comparison(results: list[dict], slots_list: list[int], bandwidth: float) -> None:
    """Hit rate and I/O time per policy, both as a delta against LFU.

    The upstream claim this checks (`docs/landscape.md`) is stated in
    ms/token, so both views are printed on the same traces.
    """
    policies = list(dict.fromkeys(r["policy"] for r in results))

    def pick(policy: str, slots: int) -> dict | None:
        return next(
            (x for x in results if x["policy"] == policy and x["slots"] == slots),
            None,
        )

    for title, fmt in (
        ("hit rate", "hit"),
        ("expert-read time per decode token", "ms"),
    ):
        print("=" * 96)
        print(f"POLICY COMPARISON: {title}  (same traces, delta vs LFU)")
        print("=" * 96)
        header = f"{'slots':>5} " + " ".join(f"{p:>18}" for p in policies)
        print(header)
        print("-" * len(header))
        for slots in slots_list:
            row = [f"{slots:>5} "]
            base = pick("lfu", slots)
            for policy in policies:
                r = pick(policy, slots)
                if r is None:
                    row.append(f"{'-':>18} ")
                    continue
                if fmt == "hit":
                    value = 100 * r["hit_rate"]
                    cell = f"{value:.2f}%"
                    if base is not None and policy != "lfu":
                        cell += f" ({value - 100 * base['hit_rate']:+.2f})"
                else:
                    value = 1000 * io_seconds(r["bytes_per_token"], bandwidth)
                    cell = f"{value:.1f}ms"
                    if base is not None and policy != "lfu":
                        base_ms = 1000 * io_seconds(base["bytes_per_token"], bandwidth)
                        cell += f" ({value - base_ms:+.1f})"
                row.append(f"{cell:>18} ")
            print("".join(row))
        print()


def print_knee(results: list[dict], bandwidth: float, policy: str) -> list[dict]:
    """Marginal value of each extra slot: the knee of the curve."""
    print("=" * 96)
    print(f"MARGINAL VALUE OF MEMORY  (policy {policy})")
    print("=" * 96)
    header = (
        f"{'slots':>5} {'pool MiB':>9} {'hit %':>7} {'io ms/tok':>10} "
        f"{'d(hit %)':>9} {'d(MiB)':>8} {'hit % per 100 MiB':>18} {'ms/tok saved per 100 MiB':>25}"
    )
    print(header)
    print("-" * len(header))
    rows = sorted((r for r in results if r["policy"] == policy), key=lambda r: r["slots"])
    out = []
    prev = None
    for r in rows:
        secs = io_seconds(r["bytes_per_token"], bandwidth)
        entry = {
            "slots": r["slots"],
            "pool_mib": mib(r["memory_bytes"]),
            "hit_rate": r["hit_rate"],
            "io_ms_per_token": 1000 * secs,
        }
        if prev is None:
            print(
                f"{r['slots']:>5} {entry['pool_mib']:>9.0f} {100 * r['hit_rate']:>7.2f} "
                f"{entry['io_ms_per_token']:>10.1f} {'-':>9} {'-':>8} {'-':>18} {'-':>25}"
            )
        else:
            d_hit = 100 * (r["hit_rate"] - prev["hit_rate"])
            d_mib = entry["pool_mib"] - prev["pool_mib"]
            d_ms = prev["io_ms_per_token"] - entry["io_ms_per_token"]
            entry["hit_pct_per_100mib"] = 100 * d_hit / d_mib if d_mib else 0.0
            entry["ms_saved_per_100mib"] = 100 * d_ms / d_mib if d_mib else 0.0
            print(
                f"{r['slots']:>5} {entry['pool_mib']:>9.0f} {100 * r['hit_rate']:>7.2f} "
                f"{entry['io_ms_per_token']:>10.1f} {d_hit:>9.2f} {d_mib:>8.0f} "
                f"{entry['hit_pct_per_100mib']:>18.2f} {entry['ms_saved_per_100mib']:>25.1f}"
            )
        prev = entry
        out.append(entry)
    print()
    return out


def print_layer_profile(result: dict, stats: dict) -> None:
    print("=" * 96)
    print(
        f"PER-LAYER PROFILE  (policy {result['policy']}, {result['slots']} slots/layer)"
    )
    print("=" * 96)
    header = (
        f"{'layer':>5} {'stride KiB':>11} {'hit %':>7} {'cold miss %':>12} "
        f"{'evict miss %':>13} {'distinct':>9} {'entropy':>8} "
        f"{'top8 %':>7} {'top16 %':>8} {'consec %':>9} {'inf-cache %':>12}"
    )
    print(header)
    print("-" * len(header))
    for p in stats["per_layer"]:
        layer = p["layer"]
        cold = result["layer_cold_misses"][layer]
        evict = result["layer_eviction_misses"][layer]
        total_miss = (cold + evict) or 1
        print(
            f"{layer:>5} {p['stride'] / 1024:>11.0f} "
            f"{100 * result['layer_hit_rate'][layer]:>7.2f} "
            f"{100 * cold / total_miss:>12.1f} {100 * evict / total_miss:>13.1f} "
            f"{p['distinct_experts']:>9} "
            f"{p['entropy_bits']:>8.2f} {100 * p['top8_coverage']:>7.1f} "
            f"{100 * p['top16_coverage']:>8.1f} {100 * p['consecutive_reuse']:>9.1f} "
            f"{100 * p['infinite_cache_hit_rate']:>12.1f}"
        )
    lr = sorted(result["layer_hit_rate"])
    print(
        f"\nlayer hit-rate spread: min {100 * lr[0]:.1f}%  p25 {100 * quantile(lr, 0.25):.1f}%  "
        f"median {100 * quantile(lr, 0.5):.1f}%  p75 {100 * quantile(lr, 0.75):.1f}%  "
        f"max {100 * lr[-1]:.1f}%"
    )
    early = result["layer_hit_rate"][: len(lr) // 4]
    late = result["layer_hit_rate"][-(len(lr) // 4) :]
    print(
        f"first quarter of layers {100 * sum(early) / len(early):.1f}%  vs  "
        f"last quarter {100 * sum(late) / len(late):.1f}%"
    )
    print()


def print_routing(stats: dict) -> None:
    print("=" * 96)
    print("ROUTING BEHAVIOUR  (decode records only)")
    print("=" * 96)
    n = stats["n_experts"]
    print(f"decode tokens simulated       : {stats['decode_tokens']}")
    print(f"experts / layer, top_k        : {n}, {stats['top_k']}")
    print(
        f"distinct experts used per layer: {stats['mean_distinct_experts']:.1f} of {n} "
        f"(mean over layers)"
    )
    print(
        f"router entropy                 : {stats['mean_entropy_bits']:.2f} bits "
        f"(uniform would be {stats['uniform_entropy_bits']:.2f})"
    )
    print(
        f"top-8 / top-16 / top-32 coverage: "
        f"{100 * stats['mean_top8_coverage']:.1f}% / "
        f"{100 * stats['mean_top16_coverage']:.1f}% / "
        f"{100 * stats['mean_top32_coverage']:.1f}%"
    )
    print(
        f"consecutive-token reuse        : {100 * stats['consecutive_reuse']:.1f}% "
        f"(routed experts also routed by the previous token)"
    )
    print(
        f"infinite cache, whole trace    : {100 * stats['infinite_cache_hit_rate']:.1f}%"
    )
    print(
        f"infinite cache, after {stats['warmup_tokens']:>3} tokens: "
        f"{100 * stats['infinite_cache_hit_rate_after_warmup']:.1f}%   "
        f"(the ceiling any eviction policy is chasing)"
    )
    print()


def print_cold_start(cold: dict, slots: int, policy: str, n_traces: int) -> None:
    print("=" * 96)
    print(f"COLD START  (policy {policy}, {slots} slots/layer)")
    print("=" * 96)
    print(
        f"averaged over all {n_traces} generations, each starting from an empty cache, "
        f"truncated to the shortest ({cold['limit']} decode tokens)"
    )
    print(f"{'tokens':>14} {'window hit %':>13} {'cumulative hit %':>18}")
    print("-" * 47)
    for w, c in zip(cold["windows"], cold["cumulative"]):
        print(
            f"{w['from_token']:>6}-{w['to_token']:<7} {100 * w['hit_rate']:>13.2f} "
            f"{100 * c['hit_rate']:>18.2f}"
        )
    steady = 100 * cold["steady_state"]
    settle = cold["tokens_to_steady"]
    print(
        f"\nsteady state (last half of decode): {steady:.2f}%; reached within "
        f"{100 * cold['tolerance']:.0f} points after "
        f"{'never' if settle is None else str(settle) + ' tokens'}"
    )
    print()


def print_aging_sweep(rows: list[dict], baseline: dict, slots: int) -> None:
    """How sensitive `lfu-aged` is to the decay period, against plain LFU."""
    print("=" * 96)
    print(f"LFU AGING SENSITIVITY  ({slots} slots/layer)")
    print("=" * 96)
    print(f"{'halve every N accesses':>24} {'~tokens':>9} {'hit %':>8} {'vs plain LFU':>14}")
    print("-" * 58)
    print(
        f"{'never (plain LFU)':>24} {'-':>9} {100 * baseline['hit_rate']:>8.2f} {'-':>14}"
    )
    for r in rows:
        delta = 100 * (r["hit_rate"] - baseline["hit_rate"])
        print(
            f"{r['age_period']:>24} {r['age_period'] / 8:>9.0f} "
            f"{100 * r['hit_rate']:>8.2f} {delta:>+14.2f}"
        )
    print()


def print_pool(rows: list[dict], policy: str) -> None:
    """Uniform per-layer slots vs the best static split of the same total."""
    print("=" * 96)
    print(f"GLOBAL POOL vs PER-LAYER SLOTS  (policy {policy}, same total memory)")
    print("=" * 96)
    header = (
        f"{'slots/layer':>12} {'total slots':>12} {'uniform hit %':>14} "
        f"{'best split hit %':>17} {'gain':>7} {'slots min..max':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['slots']:>12} {r['total_slots']:>12} {100 * r['uniform']:>14.2f} "
            f"{100 * r['pooled']:>17.2f} {100 * (r['pooled'] - r['uniform']):>+7.2f} "
            f"{r['alloc_min']:>7}..{r['alloc_max']:<7}"
        )
    print()


def print_prefill_warm(cold_res: dict, warm_res: dict, bandwidth: float) -> None:
    print("=" * 96)
    print("PREFILL WARMING  (does replaying the prompt into the cache pay?)")
    print("=" * 96)
    for label, r in (("cold after prefill", cold_res), ("warmed by prefill", warm_res)):
        secs = io_seconds(r["bytes_per_token"], bandwidth)
        print(
            f"{label:<20} hit {100 * r['hit_rate']:>6.2f}%   "
            f"{r['bytes_per_token'] / 1e6:>7.1f} MB/tok   {1000 * secs:>7.1f} io ms/tok"
        )
    delta = 100 * (warm_res["hit_rate"] - cold_res["hit_rate"])
    print(f"\ndelta from warming: {delta:+.2f} points\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate the per-layer expert cache over captured routing traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--trace",
        action="append",
        nargs="+",
        required=True,
        metavar="FILE",
        help="routing traces from `ramvamp generate --trace-experts` "
        "(repeatable, and each takes a shell glob)",
    )
    p.add_argument(
        "--layout",
        default="models/qwen3.rvmp/experts/layout.json",
        help="install layout.json, for the real per-layer strides",
    )
    p.add_argument(
        "--slots",
        default=",".join(str(s) for s in DEFAULT_SLOTS),
        help="comma-separated slots-per-layer sweep",
    )
    p.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help=f"comma-separated eviction policies from {sorted(POLICIES)}",
    )
    p.add_argument(
        "--age-period",
        type=int,
        default=DEFAULT_AGE_PERIOD,
        help="accesses per layer between LFU counter halvings (lfu-aged)",
    )
    p.add_argument(
        "--window-tokens",
        type=int,
        default=DEFAULT_WINDOW_TOKENS,
        help="frequency window in decode tokens for lfu-window",
    )
    p.add_argument(
        "--bandwidth",
        type=float,
        default=DEFAULT_BANDWIDTH_GBPS,
        help="expert-read bandwidth in GB/s used for the I/O projections",
    )
    p.add_argument(
        "--profile-slots",
        type=int,
        default=10,
        help="slots/layer used for the per-layer and cold-start sections",
    )
    p.add_argument("--window", type=int, default=16, help="cold-start window in tokens")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="hit-rate band (fraction) that counts as steady state",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=32,
        help="decode tokens skipped for the post-warmup infinite-cache figure",
    )
    p.add_argument(
        "--no-pool-analysis",
        dest="pool_analysis",
        action="store_false",
        help="skip the per-layer hit curves and the global-pool comparison "
        "(the slowest section)",
    )
    p.add_argument("--out", metavar="FILE", help="write the full results as JSON here")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    slots_list = sorted({int(s) for s in args.slots.split(",") if s.strip()})
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for policy in policies:
        if policy not in POLICIES:
            fail(f"unknown policy {policy!r}; known: {sorted(POLICIES)}")

    strides = read_strides(args.layout)
    paths = sorted({path for group in args.trace for path in group})
    traces = [read_trace(path) for path in paths]
    if not traces:
        fail("no traces given")
    for t in traces[1:]:
        if (t.n_layers, t.n_experts, t.top_k) != (
            traces[0].n_layers,
            traces[0].n_experts,
            traces[0].top_k,
        ):
            fail(f"{t.name}: geometry differs from {traces[0].name}")
    if len(strides) != traces[0].n_layers:
        fail(f"layout has {len(strides)} layers, traces have {traces[0].n_layers}")

    print("=" * 96)
    print("EXPERT CACHE SIMULATION")
    print("=" * 96)
    print(f"layout        : {args.layout}")
    print(
        f"strides       : {min(strides):,} B .. {max(strides):,} B per expert, "
        f"{sum(strides) / 1e6:.1f} MB for one slot across all {len(strides)} layers"
    )
    print(f"bandwidth     : {args.bandwidth} GB/s")
    total_decode = 0
    for t in traces:
        n_decode = len(t.records_of(PHASE_DECODE))
        n_prefill = len(t.records_of(PHASE_PREFILL))
        total_decode += n_decode
        print(f"trace         : {t.name}  {n_prefill} prefill + {n_decode} decode records")
    print(f"decode tokens : {total_decode}")
    no_cache = sum(
        strides[layer] * traces[0].top_k for layer in range(traces[0].n_layers)
    )
    secs = io_seconds(no_cache, args.bandwidth)
    print(
        f"no cache      : {no_cache / 1e6:.1f} MB/tok, {1000 * secs:.1f} io ms/tok, "
        f"{1 / secs:.2f} io-only tok/s"
    )

    results = []
    for policy in policies:
        for slots in slots_list:
            results.append(
                simulate(
                    traces,
                    strides,
                    slots,
                    policy,
                    args.age_period,
                    False,
                    args.window_tokens,
                )
            )

    stats = routing_stats(traces, strides, args.warmup)
    print_routing(stats)
    print_sweep(results, args.bandwidth, slots_list)
    print_policy_comparison(results, slots_list, args.bandwidth)
    reference = print_reference(results, policies)
    # Everything below profiles the best *implementable* policy (Belady is
    # a bound, not a candidate), judged at the profile slot count.
    online = [p for p in policies if p != OptCache.label] or policies
    knee_policy = max(
        online,
        key=lambda p: sum(r["hit_rate"] for r in results if r["policy"] == p),
    )
    print(f"profiling the best implementable policy: {knee_policy}\n")
    budget = print_budget(results, slots_list, knee_policy)
    print_misses(results, slots_list, knee_policy)
    knee = print_knee(results, args.bandwidth, knee_policy)

    profile = next(
        (
            r
            for r in results
            if r["policy"] == knee_policy and r["slots"] == args.profile_slots
        ),
        None,
    )
    if profile is None:
        profile = simulate(
            traces,
            strides,
            args.profile_slots,
            knee_policy,
            args.age_period,
            False,
            args.window_tokens,
        )
        results.append(profile)
    print_layer_profile(profile, stats)
    shortest = min(len(t.records_of(PHASE_DECODE)) for t in traces)
    cold = cold_start(profile, args.window, args.tolerance, shortest)
    print_cold_start(cold, profile["slots"], profile["policy"], len(traces))

    aging = []
    if "lfu-aged" in policies:
        for period in (32, 64, 128, 256, 512, 1024):
            r = simulate(
                traces, strides, args.profile_slots, "lfu-aged", period, False
            )
            r["age_period"] = period
            aging.append(
                {k: v for k, v in r.items() if not k.startswith("per_token_")}
            )
        lfu_base = next(
            (r for r in results if r["policy"] == "lfu" and r["slots"] == args.profile_slots),
            profile,
        )
        print_aging_sweep(aging, lfu_base, args.profile_slots)

    pool_rows: list[dict] = []
    if args.pool_analysis:
        # Headroom above the sweep so the allocator can actually skew.
        slots_max = min(2 * max(slots_list), traces[0].n_experts)
        curves, accesses = layer_curves(traces, slots_max, knee_policy, args.age_period)
        n_layers = traces[0].n_layers
        for slots in slots_list:
            alloc, hits = greedy_pool(curves, slots * n_layers)
            uniform = sum(c[slots - 1] for c in curves) / accesses
            pool_rows.append(
                {
                    "slots": slots,
                    "total_slots": slots * n_layers,
                    "uniform": uniform,
                    "pooled": hits / accesses,
                    "alloc_min": min(alloc),
                    "alloc_max": max(alloc),
                    "alloc": alloc,
                }
            )
        print_pool(pool_rows, knee_policy)

    warm = simulate(
        traces,
        strides,
        args.profile_slots,
        knee_policy,
        args.age_period,
        True,
        args.window_tokens,
    )
    print_prefill_warm(profile, warm, args.bandwidth)

    if args.out:
        payload = {
            "layout": os.path.abspath(args.layout),
            "strides": strides,
            "slot_pool_bytes_per_slot": sum(strides),
            "bandwidth_gbps": args.bandwidth,
            "age_period": args.age_period,
            "traces": [
                {
                    "name": t.name,
                    "n_layers": t.n_layers,
                    "n_experts": t.n_experts,
                    "top_k": t.top_k,
                    "prefill_records": len(t.records_of(PHASE_PREFILL)),
                    "decode_records": len(t.records_of(PHASE_DECODE)),
                }
                for t in traces
            ],
            "no_cache_bytes_per_token": no_cache,
            "profile_policy": knee_policy,
            "upstream_reference": reference,
            "memory_budget": budget,
            "routing": stats,
            "sweep": [
                {
                    k: v
                    for k, v in r.items()
                    if k
                    not in (
                        "per_token_hits",
                        "per_token_accesses",
                        "layer_reuse_histogram",
                    )
                }
                | {
                    "memory_mib": mib(r["memory_bytes"]),
                    "io_seconds_per_token": io_seconds(
                        r["bytes_per_token"], args.bandwidth
                    ),
                    "io_only_tok_per_s": (
                        1 / io_seconds(r["bytes_per_token"], args.bandwidth)
                        if r["bytes_per_token"]
                        else None
                    ),
                }
                for r in results
            ],
            "knee": knee,
            "aging_sensitivity": aging,
            "global_pool": pool_rows,
            "cold_start": cold,
            "prefill_warming": {
                "slots": args.profile_slots,
                "policy": knee_policy,
                "cold_hit_rate": profile["hit_rate"],
                "warm_hit_rate": warm["hit_rate"],
                "cold_bytes_per_token": profile["bytes_per_token"],
                "warm_bytes_per_token": warm["bytes_per_token"],
            },
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
