#!/usr/bin/env python3
"""Multi-position decode regression vs the banked llama.cpp fixtures.

Everything else in the repo validates *position 0*: `kl_vs_reference.py`
compares one distribution after one prefill, `bitident.py` fingerprints
the same. Phase 5 (io_uring + O_DIRECT + per-layer expert cache + pinned
compute threads) can be perfect at position 0 and still be wrong at
position 37 — a cache that returns a stale expert, a completion that
lands after the consumer read the buffer, or a KV write racing a read all
produce a correct first token and garbage afterwards. This script is the
gate for that failure mode. It consumes the two fixtures under
`models/llamacpp-ref/llamacpp_ref/` that nothing else reads:

  greedy_texts.json  128-token greedy continuations of the 8 single
                     prompts (llama.cpp b10217, temperature 0). Metric:
                     how many leading tokens of our greedy continuation
                     are byte-identical to theirs. This exercises the
                     *decode* loop end to end, including the KV cache.

  path_*.npz         3 prompts x 64 decode positions x top-5000 logprobs.
                     Metric: teacher-forced per-position agreement — we
                     re-run `logits` on prompt + the reference's own
                     chosen tokens, so both sides sit at an identical
                     context at every position, and compare top-1 / top-5
                     / top-10.

## Why the pass criterion is "no worse than the phase-4 baseline"

These are llama.cpp numbers, and exact agreement is not achievable:
EXP-004 measured mean full-vocab KL 1.04e-2 against a 0.5-1.3e-2
*intra-engine* noise floor (same ramvamp binary, AVX2 vs forced scalar).
Two engines that do not replicate float accumulation operation for
operation land O(1e-2) apart per position, which is enough to flip the
argmax whenever the top two candidates are close, and one flipped argmax
sends a greedy continuation somewhere else entirely. So "greedy text must
match all 128 tokens" and "top-1 must agree at all 64 positions" are both
wrong gates — phase 4, which is known good, fails them (see the measured
baseline below).

What *is* meaningful is that the numbers do not move. The divergence
point of a greedy continuation and the per-position top-1 agreement count
are deterministic functions of the arithmetic; a change in either means
the forward pass changed. So this script records the phase-4 measurement
as `baseline.json` and fails if any per-prompt or per-path number drops
below it. Ties (a number that goes *up*) pass, and are reported — an
improvement is still a change and should be explained, but it is not a
regression, and holding out for exact equality would make the gate
brittle against harmless top-k reshuffling deep in the tail.

`bitident.py` is the sharper gate (byte-identical logits, no tolerance at
all); this one is the semantic backstop that says *where* in a sequence a
change first shows up.

## Measured phase-4 baseline (2026-08-03, commit 650b5ea + scripts,
## release build, 185H, warm page cache — this is a correctness
## measurement, cgroup/cold rules do not apply). Recorded in
## `<cache>/baseline.json`.

Greedy, `--max-new 128` (leading tokens matching llama.cpp exactly):

  single_00  The capital of France is                     17/128
  single_01  Water is composed of                         52/128
  single_02  In Rust, ownership means                     27/128
  single_03  Once upon a time, in a village by the sea,   19/128
  single_04  The derivative of x^2 with respect to x is  128/128  (full)
  single_05  def fibonacci(n):                            13/128
  single_06  The three primary colors are                114/128
  single_07  Photosynthesis is the process by which       37/128
  aggregate  407/1024 tokens = 39.7%, 1/8 prompts identical for all 128

Teacher-forced paths, `--stride 8` (8 of 64 positions per path, top 64):

  path_00  top-1 8/8  top-5 37/40  top-10 76/80  first disagreement none
  path_01  top-1 8/8  top-5 38/40  top-10 77/80  first disagreement none
  path_02  top-1 8/8  top-5 35/40  top-10 76/80  first disagreement none
  aggregate top-1 24/24 = 100%

  Retokenization check: 24/24 contexts re-encoded to exactly the
  reference's token ids, so every comparison is at an identical context.

Read those two blocks together — they are the argument for why the gate
is phrased as "no worse than baseline" rather than "must agree":

  * Teacher-forced, ramvamp's argmax matches llama.cpp's at **every**
    sampled position, out to position 56 at context depth 61. Nothing is
    wrong with the forward pass, the KV cache or RoPE at depth.
  * Free-running, the same engine drops to 39.7% of banked greedy tokens.
    The two facts are consistent: one flipped argmax at a near-tie is
    enough to send the continuation somewhere else and never come back,
    and at O(1e-2) per-position KL (EXP-004) near-ties flip. 7 of 8
    prompts diverge, at token 13 through 114 — the divergence point is a
    stable fingerprint of the arithmetic, not a quality signal.

So `matched_tokens` should be read as "the point at which accumulated
float noise first tipped a decision", and any *movement* in it means the
arithmetic changed. Phase 5 must not move it.

Full `--stride 1` costs ~50 min per path at phase-4 speeds (~1.3-1.9 s
per prefill token, O(n^2) because `logits` re-prefills from scratch); the
stride-8 baseline above took ~20 min for all three paths and ~29 min for
the 8 greedy continuations (both suites run concurrently on a
memory-pressured box). Phase 5 should make `--stride 1` affordable —
re-record the baseline with `--save-baseline` if you widen it, since the
gate refuses to compare across configurations.

## Usage

  scripts/greedy_regression.py --ramvamp target/release/ramvamp
  scripts/greedy_regression.py --refresh --ramvamp target/release/ramvamp
  scripts/greedy_regression.py --only paths --stride 1 --refresh

Results are cached per (prompt, position) under `--cache`, so a rerun
with the same configuration is free; `--refresh` recomputes, which is
what a phase-5 wave must pass. Exit 0 = PASS, 1 = FAIL, 2 = harness error.

Python stdlib only.
"""

from __future__ import annotations

import argparse
import array
import ast
import io
import json
import math
import os
import struct
import subprocess
import sys
import time
import zipfile

# Depth of the per-position top-k dump. Only affects JSON size: `logits`
# sorts the whole 151936-token vocabulary regardless. 64 is deep enough to
# report where the reference's argmax landed on our side when it is not
# our argmax.
DEFAULT_PATH_TOP = 64


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# minimal .npz reader (same subset as scripts/kl_vs_reference.py)
# ---------------------------------------------------------------------------


def read_npy(fh) -> tuple[tuple[int, ...], array.array]:
    if fh.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = fh.read(2)[0]
    if major == 1:
        (hlen,) = struct.unpack("<H", fh.read(2))
    else:
        (hlen,) = struct.unpack("<I", fh.read(4))
    header = ast.literal_eval(fh.read(hlen).decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("fortran_order arrays not supported")
    codes = {"<i4": "i", "<u4": "I", "<i8": "q", "<f4": "f", "<f8": "d"}
    descr = header["descr"]
    if descr not in codes:
        raise ValueError(f"unsupported dtype {descr!r}")
    arr = array.array(codes[descr])
    count = math.prod(header["shape"])
    arr.frombytes(fh.read(count * arr.itemsize))
    if len(arr) != count:
        raise ValueError("truncated .npy payload")
    if sys.byteorder == "big":
        arr.byteswap()
    return header["shape"], arr


def read_npz(path: str) -> dict[str, tuple[tuple[int, ...], array.array]]:
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as fh:
                out[name.removesuffix(".npy")] = read_npy(io.BytesIO(fh.read()))
    return out


# ---------------------------------------------------------------------------
# ByteLevel BPE detokenizer (ids -> bytes) straight from the model's
# tokenizer.json. Needed because the fixtures store token ids and we need
# the exact byte boundaries between them; `ramvamp` has no decode-ids CLI.
# Verified against greedy_texts.json: decode(tokens) == content, 8/8.
# ---------------------------------------------------------------------------


def byte_level_map() -> dict[str, int]:
    """The GPT-2 printable-unicode <-> byte table used by ByteLevel."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    codes = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            codes.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, codes)}


class Detokenizer:
    def __init__(self, model_dir: str):
        path = os.path.join(model_dir, "tokenizer", "tokenizer.json")
        if not os.path.isfile(path):
            fail(f"no tokenizer.json under {model_dir}")
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        if spec.get("decoder", {}).get("type") != "ByteLevel":
            fail(f"unexpected decoder {spec.get('decoder')!r}; this reader "
                 f"only implements ByteLevel")
        self.pieces = {v: k for k, v in spec["model"]["vocab"].items()}
        for added in spec.get("added_tokens", []):
            self.pieces[added["id"]] = added["content"]
        self.bytes_of = byte_level_map()

    def decode(self, ids) -> bytes:
        out = bytearray()
        for tid in ids:
            piece = self.pieces.get(int(tid))
            if piece is None:
                fail(f"token id {tid} is not in the tokenizer vocab")
            out.extend(self.bytes_of[c] for c in piece)
        return bytes(out)


# ---------------------------------------------------------------------------
# ramvamp side (cached)
# ---------------------------------------------------------------------------


def base_cmd(args) -> list[str]:
    if args.ramvamp:
        return [args.ramvamp]
    return ["cargo", "run", "--release", "--quiet", "-p", "ramvamp", "--"]


def cached(args, cache_path: str, produce) -> dict:
    if os.path.isfile(cache_path) and not args.refresh:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    payload = produce()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def run(args, cmd: list[str], what: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, check=False
        )
    except subprocess.TimeoutExpired:
        fail(f"ramvamp timed out after {args.timeout:.0f}s on {what}")
    if result.returncode != 0:
        fail(f"ramvamp exited {result.returncode} on {what}\n"
             f"stderr:\n{result.stderr[-2000:]}")
    return result


def ramvamp_generate(args, prompt: str, cache_path: str) -> dict:
    def produce() -> dict:
        cmd = base_cmd(args) + [
            "generate",
            "--model", args.rvmp,
            "--prompt", prompt,
            "--greedy",
            "--max-new", str(args.max_new),
            "--skip-hashes",
        ]
        print(f"  + generate --max-new {args.max_new} {prompt[:48]!r}",
              file=sys.stderr)
        result = run(args, cmd, f"generate {prompt[:60]!r}")
        # `generate` streams the continuation to stdout and then prints one
        # bare newline as a terminator; strip exactly that one.
        text = result.stdout
        if text.endswith("\n"):
            text = text[:-1]
        return {"prompt": prompt, "text": text, "stderr": result.stderr[-4000:]}

    return cached(args, cache_path, produce)


def ramvamp_logits(args, prompt: str, cache_path: str, top: int) -> dict:
    def produce() -> dict:
        cmd = base_cmd(args) + [
            "logits",
            "--model", args.rvmp,
            "--prompt", prompt,
            "--top", str(top),
            "--skip-hashes",
        ]
        result = run(args, cmd, f"logits ({len(prompt)} chars)")
        return json.loads(result.stdout)

    return cached(args, cache_path, produce)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def matched_token_prefix(detok: Detokenizer, ref_ids, ours: bytes) -> tuple[int, dict]:
    """How many leading reference tokens our byte stream reproduces exactly."""
    cum = bytearray()
    for k, tid in enumerate(ref_ids):
        cum.extend(detok.decode([tid]))
        if ours[: len(cum)] != bytes(cum):
            start = len(cum) - len(detok.decode([tid]))
            return k, {
                "diverged_at_token": k,
                "diverged_at_byte": start,
                "ref_next": detok.decode(ref_ids[k : k + 6]).decode(
                    "utf-8", "replace"),
                "our_next": ours[start : start + 32].decode("utf-8", "replace"),
            }
    return len(ref_ids), {}


def topk_agreement(ref_ids: list[int], ours: list[int]) -> dict:
    """top-1 / top-5 / top-10 overlap plus where our list puts their argmax."""
    ref5, ref10 = set(ref_ids[:5]), set(ref_ids[:10])
    our5, our10 = set(ours[:5]), set(ours[:10])
    try:
        rank = ours.index(ref_ids[0])
    except ValueError:
        rank = -1
    return {
        "top1_agree": bool(ref_ids[0] == ours[0]),
        "top5_overlap": len(ref5 & our5),
        "top10_overlap": len(ref10 & our10),
        "ref_top1": int(ref_ids[0]),
        "our_top1": int(ours[0]),
        "rank_of_ref_top1": rank,
    }


# ---------------------------------------------------------------------------
# the two suites
# ---------------------------------------------------------------------------


def run_greedy(args, meta: dict, detok: Detokenizer) -> dict:
    ref_path = os.path.join(args.ref, "greedy_texts.json")
    if not os.path.isfile(ref_path):
        fail(f"no greedy_texts.json under {args.ref}")
    with open(ref_path, encoding="utf-8") as f:
        banked = json.load(f)
    by_prompt = {e["prompt"]: e for e in banked}

    per_prompt = []
    for entry in meta["singles"]:
        name, prompt = entry["file"], entry["prompt"]
        ref = by_prompt.get(prompt)
        if ref is None:
            print(f"[{name}] SKIP: no banked greedy text", file=sys.stderr)
            continue
        want = min(args.max_new, len(ref["tokens"]))
        t0 = time.time()
        got = ramvamp_generate(
            args, prompt,
            os.path.join(args.cache, f"greedy_{name}_n{args.max_new}.json"))
        ours = got["text"].encode("utf-8")
        matched, detail = matched_token_prefix(detok, ref["tokens"][:want], ours)
        row = {
            "file": name, "prompt": prompt,
            "reference_tokens": want, "matched_tokens": matched,
            **detail,
        }
        per_prompt.append(row)
        tag = "full" if matched == want else f"diverges at {matched}"
        print(f"[{name}] greedy {matched:>4}/{want} tokens match ({tag})  "
              f"({time.time() - t0:.0f}s)  {prompt!r}")
        if detail:
            print(f"          ref {detail['ref_next']!r}")
            print(f"          our {detail['our_next']!r}")

    total = sum(r["matched_tokens"] for r in per_prompt)
    possible = sum(r["reference_tokens"] for r in per_prompt)
    full = sum(1 for r in per_prompt if r["matched_tokens"] == r["reference_tokens"])
    print(f"\ngreedy aggregate: {total}/{possible} tokens "
          f"({100.0 * total / possible if possible else 0.0:.1f}%), "
          f"{full}/{len(per_prompt)} prompts identical for all "
          f"{args.max_new} tokens")
    return {
        "max_new": args.max_new,
        "per_prompt": per_prompt,
        "matched_total": total,
        "possible_total": possible,
        "full_match_prompts": full,
    }


def run_paths(args, meta: dict, detok: Detokenizer) -> dict:
    ids_by_prompt = {e["prompt"]: list(e["ids"])
                     for e in meta.get("tokenized_prompts", [])}
    per_path = []
    for entry in meta.get("paths", []):
        name, prompt = entry["file"], entry["prompt"]
        npz_path = os.path.join(args.ref, f"{name}.npz")
        if not os.path.isfile(npz_path):
            print(f"[{name}] SKIP: no {name}.npz", file=sys.stderr)
            continue
        arrays = read_npz(npz_path)
        (n_pos,), chosen = arrays["chosen_ids"]
        (rows, depth), top_ids = arrays["top_ids"]
        if rows != n_pos:
            fail(f"{name}.npz: {rows} top-k rows but {n_pos} chosen ids")
        prompt_ids = ids_by_prompt.get(prompt)

        positions = list(range(0, n_pos, args.stride))
        if args.positions:
            positions = positions[: args.positions]

        rows_out = []
        first_disagree = None
        retokenized = 0
        t_path = time.time()
        for pos in positions:
            context = prompt + detok.decode(chosen[:pos]).decode("utf-8", "replace")
            rv = ramvamp_logits(
                args, context,
                os.path.join(args.cache,
                             f"{name}_p{pos:03d}_top{args.top}.json"),
                args.top)
            ours = [row["token_id"] for row in rv["top"]]
            ref_row = [int(t) for t in top_ids[pos * depth : pos * depth + 10]]
            agree = topk_agreement(ref_row, ours)

            ctx_ok = True
            if prompt_ids is not None:
                want_ids = prompt_ids + [int(t) for t in chosen[:pos]]
                ctx_ok = list(rv["prompt_ids"]) == want_ids
                if not ctx_ok:
                    retokenized += 1
            agree["position"] = pos
            agree["context_ids_match"] = ctx_ok
            rows_out.append(agree)
            if not agree["top1_agree"] and first_disagree is None:
                first_disagree = pos

        n = len(rows_out)
        t1 = sum(r["top1_agree"] for r in rows_out)
        t5 = sum(r["top5_overlap"] for r in rows_out)
        t10 = sum(r["top10_overlap"] for r in rows_out)
        per_path.append({
            "file": name, "prompt": prompt, "positions": n,
            "stride": args.stride, "top": args.top,
            "top1_agree": t1, "top5_overlap": t5, "top10_overlap": t10,
            "first_disagreement": first_disagree,
            "retokenized_positions": retokenized,
            "per_position": rows_out,
        })
        where = "none" if first_disagree is None else f"pos {first_disagree}"
        print(f"[{name}] {n} positions (stride {args.stride})  "
              f"top-1 {t1}/{n}  top-5 {t5}/{5 * n}  top-10 {t10}/{10 * n}  "
              f"first disagreement {where}  "
              f"retokenized {retokenized}/{n}  ({time.time() - t_path:.0f}s)")

    n_tot = sum(p["positions"] for p in per_path)
    t1_tot = sum(p["top1_agree"] for p in per_path)
    print(f"\npath aggregate: top-1 {t1_tot}/{n_tot} "
          f"({100.0 * t1_tot / n_tot if n_tot else 0.0:.1f}%)")
    return {
        "stride": args.stride, "top": args.top,
        "per_path": per_path,
        "positions_total": n_tot,
        "top1_total": t1_tot,
    }


# ---------------------------------------------------------------------------
# baseline gate
# ---------------------------------------------------------------------------


def gate(results: dict, baseline: dict) -> tuple[bool, list[str]]:
    """No measured number may drop below the recorded phase-4 baseline."""
    problems = []

    old, new = baseline.get("greedy"), results.get("greedy")
    if old and new:
        if old["max_new"] != new["max_new"]:
            problems.append(
                f"greedy not gated: baseline --max-new {old['max_new']}, "
                f"this run {new['max_new']}")
        else:
            was = {r["file"]: r["matched_tokens"] for r in old["per_prompt"]}
            for row in new["per_prompt"]:
                before = was.get(row["file"])
                if before is None:
                    continue
                if row["matched_tokens"] < before:
                    problems.append(
                        f"greedy {row['file']}: matched {row['matched_tokens']} "
                        f"tokens, baseline {before} (-{before - row['matched_tokens']})")
            if new["matched_total"] < old["matched_total"]:
                problems.append(
                    f"greedy aggregate {new['matched_total']}, baseline "
                    f"{old['matched_total']}")

    old, new = baseline.get("paths"), results.get("paths")
    if old and new:
        if (old["stride"], old["top"]) != (new["stride"], new["top"]):
            problems.append(
                f"paths not gated: baseline stride {old['stride']}/top "
                f"{old['top']}, this run stride {new['stride']}/top {new['top']}")
        else:
            was = {p["file"]: p for p in old["per_path"]}
            for path in new["per_path"]:
                before = was.get(path["file"])
                if before is None:
                    continue
                for key in ("top1_agree", "top5_overlap", "top10_overlap"):
                    if path[key] < before[key]:
                        problems.append(
                            f"{path['file']}: {key} {path[key]}, baseline "
                            f"{before[key]}")
                b_first = before["first_disagreement"]
                n_first = path["first_disagreement"]
                if b_first is None and n_first is not None:
                    problems.append(
                        f"{path['file']}: top-1 now disagrees at position "
                        f"{n_first}, baseline agreed everywhere")
                elif (b_first is not None and n_first is not None
                      and n_first < b_first):
                    problems.append(
                        f"{path['file']}: first top-1 disagreement moved "
                        f"earlier, {b_first} -> {n_first}")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rvmp", default="models/qwen3.rvmp",
                        help="installed .rvmp model dir")
    parser.add_argument("--ref", default="models/llamacpp-ref/llamacpp_ref",
                        help="reference dump dir (meta.json, greedy_texts.json, "
                             "path_*.npz)")
    parser.add_argument("--ramvamp",
                        help="ramvamp binary (default: cargo run --release)")
    parser.add_argument("--cache", default="models/llamacpp-ref/greedy-cache",
                        help="where per-(prompt, position) ramvamp outputs live")
    parser.add_argument("--refresh", action="store_true",
                        help="recompute every cached ramvamp output; this is "
                             "what a phase-5 wave must run")
    parser.add_argument("--only", choices=["all", "greedy", "paths"],
                        default="all", help="run one suite only")
    parser.add_argument("--max-new", type=int, default=128,
                        help="greedy tokens per prompt (banked depth is 128)")
    parser.add_argument("--stride", type=int, default=8,
                        help="path position stride; the default matches the "
                             "recorded baseline (8 of 64 positions per path). "
                             "--stride 1 sweeps all 64 but costs ~50 min per "
                             "path at phase-4 speeds and needs the baseline "
                             "re-recorded, since the gate refuses to compare "
                             "across configurations")
    parser.add_argument("--positions", type=int,
                        help="cap the number of path positions per prompt")
    parser.add_argument("--top", type=int, default=DEFAULT_PATH_TOP,
                        help=f"path top-k depth (default {DEFAULT_PATH_TOP})")
    parser.add_argument("--baseline",
                        help="baseline JSON (default: <cache>/baseline.json)")
    parser.add_argument("--save-baseline", action="store_true",
                        help="record this run as the baseline future runs "
                             "must not regress from")
    parser.add_argument("--timeout", type=float, default=10800.0,
                        help="per-invocation ramvamp timeout (s)")
    args = parser.parse_args()
    # Runs take tens of minutes at phase-4 speeds; keep a redirected log
    # readable while it happens.
    sys.stdout.reconfigure(line_buffering=True)

    if args.stride < 1:
        fail("--stride must be >= 1")
    meta_path = os.path.join(args.ref, "meta.json")
    if not os.path.isfile(meta_path):
        fail(f"no meta.json under {args.ref}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    detok = Detokenizer(args.rvmp)
    os.makedirs(args.cache, exist_ok=True)
    baseline_path = args.baseline or os.path.join(args.cache, "baseline.json")

    results: dict = {
        "model": os.path.abspath(args.rvmp),
        "reference": os.path.abspath(args.ref),
        "llama_server_version": meta.get("llama_server_version"),
        "ran": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "refreshed": args.refresh,
    }
    if args.only in ("all", "greedy"):
        results["greedy"] = run_greedy(args, meta, detok)
        print()
    if args.only in ("all", "paths"):
        results["paths"] = run_paths(args, meta, detok)
        print()

    out_path = os.path.join(args.cache, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")

    if args.save_baseline:
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"recorded baseline -> {baseline_path}")
        print("verdict: BASELINE RECORDED (nothing to gate against yet)")
        return 0

    if not os.path.isfile(baseline_path):
        print(f"verdict: NO BASELINE at {baseline_path} — rerun with "
              f"--save-baseline to record this run as the phase-4 reference")
        return 0

    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    ok, problems = gate(results, baseline)
    for problem in problems:
        print(f"  REGRESSION: {problem}")
    if ok:
        print(f"verdict: PASS — no metric below the baseline recorded "
              f"{baseline.get('ran')}")
        return 0
    print(f"verdict: FAIL — {len(problems)} metric(s) below the baseline "
          f"recorded {baseline.get('ran')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
