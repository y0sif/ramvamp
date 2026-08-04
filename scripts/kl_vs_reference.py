#!/usr/bin/env python3
"""Full-vocab KL divergence: ramvamp vs saved llama.cpp reference dumps.

Closes validation gate 3 of docs/architecture.md (mean full-vocab KL <=
3e-2 on a fixed prompt set, revised from 1e-3 on 2026-08-03; see
KL_TARGET below) without needing llama.cpp or the GGUF on this machine. The reference side is the `single_*.npz` dumps captured from
llama-server b10217 on identical Q4_K_M bytes (see `meta.json` in the
reference directory for provenance); the ramvamp side is `ramvamp logits
--top <vocab>` run here, cached next to the reference so reruns are free.

For each prompt in the reference meta.json:

  KL(P || Q)  with P = llama.cpp, Q = ramvamp, over all 151936 tokens,
  both sides renormalized in f64 (the reference logprobs are f32 and sum
  to ~1.0003; renormalizing removes that artifact). KL(Q || P), total
  variation distance, and top-1 agreement are reported alongside.

The verdict needs all four of:

  * mean KL <= KL_TARGET (gate 3 as written in the architecture doc),
  * every individual prompt <= KL_PROMPT_CEILING — a mean over 8 prompts
    hides one blown prompt behind seven good ones,
  * top-1 agreement on every prompt — a flipped argmax is a behavioural
    change even at a KL the mean tolerates,
  * every prompt in the fixed set actually scored. A prompt skipped for a
    tokenizer mismatch or a short reference dump shrinks the gate's
    denominator, so it fails the gate instead of quietly leaving it.

Alongside the gate, and deliberately *not* part of it, every prompt's
top1-vs-top2 gap is recorded on both sides (`top1_margin_llamacpp` /
`top1_margin_ramvamp` per prompt in `kl_results.json`; the minimum is
printed in the summary). Of the four conditions above, top-1 agreement is
the only one with no recorded margin behind it: EXP-004 reports top-1 8/8
but never how close any prompt came to flipping, and it reports top-1 for
the AVX2 side only. A near-tie is therefore a more plausible source of a
future spurious FAIL than the KL ceiling is, and this measures the
distance instead of assuming it. Gaps are computed on logprobs, which is
the same as on logits: both terms of the subtraction carry the same
normalizer.

Python stdlib only: the .npz reader below covers exactly what
numpy.savez_compressed writes (v1/v2 .npy headers, little-endian scalar
dtypes, C order).

Exit codes, shared with the repo's other gate scripts (`bitident.py`,
`cold_bench.py`, `greedy_regression.py`, `lfu_sim.py`):

  0  the gate ran and passed
  1  the gate ran and failed — a real result about the code under test
  2  the gate could not run (no reference dumps, a dump that does not
     cover the full vocab, ramvamp failed to start or timed out)

Example:
  scripts/kl_vs_reference.py --rvmp models/qwen3.rvmp \
      --ref models/llamacpp-ref/llamacpp_ref
"""

from __future__ import annotations

import argparse
import array
import ast
import gzip
import io
import json
import math
import os
import struct
import subprocess
import sys
import time
import zipfile

# Architecture gate 3: mean full-vocab KL. Revised 2026-08-03 from the
# original 1e-3 to 3e-2 (docs/architecture.md "Validation protocol vs
# llama.cpp", gate 3; evidence in EXP-004). 1e-3 predates the
# implementation and is unachievable without operation-identical
# arithmetic: EXP-004 measured mean 1.04e-2 against a 0.5-1.3e-2
# *intra-engine* noise floor (same binary, AVX2 vs RAMVAMP_FORCE_SCALAR=1),
# i.e. reordering float accumulation inside one engine moves KL as much as
# the whole cross-engine gap, and scalar lands closer to llama.cpp than
# AVX2 on 2 of 3 prompts. 3e-2 sits ~3x above the measured mean;
# perplexity (gate 5) is the quality backstop.
KL_TARGET = 3e-2

# A mean is not a gate on its own: with 8 prompts, one prompt at KL 0.25
# and a flipped argmax still averages under 3e-2 behind seven good ones.
# So every prompt also carries its own ceiling, and top-1 must agree
# everywhere. Measured phase-4 spread (EXP-004 rerun, 2026-08-03): worst
# single prompt 2.72e-2 (single_00), best 2.65e-3, top-1 8/8.
#
# 6e-2 is sized against the largest float-reordering perturbation this
# codebase can produce short of an algorithmic change. EXP-004's
# scalar/AVX2 A/B (`RAMVAMP_FORCE_SCALAR=1`, same binary, dot-product
# accumulation order the only difference) moved the worst per-prompt
# *cross-engine* KL to 3.4e-2. So the ceiling sits 1.76x above the largest
# perturbation ever measured here and 2.2x above the worst status-quo
# prompt (2.72e-2): it catches a single blown prompt, and re-ordering
# every dot product in the engine is not enough to trip it.
#
# The weak spot in this gate is NOT the ceiling. It is the top-1 condition
# below, which has no recorded margin: nothing in the repo records the
# top1-vs-top2 gap on these prompts, and EXP-004 reports top-1 for the
# AVX2 side only. That is why the margin is now measured and written to
# `kl_results.json` — see `top2()` and the module docstring.
KL_PROMPT_CEILING = 6e-2


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# minimal .npz reader (stdlib)
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
# ramvamp side (cached)
# ---------------------------------------------------------------------------


def ramvamp_logits(args, prompt: str, cache_path: str, vocab: int) -> dict:
    if os.path.isfile(cache_path) and not args.refresh:
        with open(cache_path) as f:
            return json.load(f)
    if args.ramvamp:
        cmd = [args.ramvamp]
    else:
        cmd = ["cargo", "run", "--release", "--quiet", "-p", "ramvamp", "--"]
    cmd += [
        "logits",
        "--model", args.rvmp,
        "--prompt", prompt,
        "--top", str(vocab),
        "--skip-hashes",
    ]
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, check=False
        )
    except subprocess.TimeoutExpired:
        fail(f"ramvamp timed out after {args.timeout:.0f}s on {prompt[:60]!r}... "
             f"(long contexts run ~2 s/token in the phase-4 loop; raise --timeout)")
    if result.returncode != 0:
        fail(f"ramvamp exited {result.returncode}\nstderr:\n{result.stderr[-2000:]}")
    payload = json.loads(result.stdout)
    with open(cache_path, "w") as f:
        json.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def compare(p_lp: list[float], q_lp: list[float]) -> dict:
    """Both inputs are dense full-vocab logprob lists; renormalized here."""
    pz = logsumexp(p_lp)
    qz = logsumexp(q_lp)
    kl_pq = kl_qp = tv = 0.0
    for lp, lq in zip(p_lp, q_lp):
        lp -= pz
        lq -= qz
        p = math.exp(lp)
        q = math.exp(lq)
        kl_pq += p * (lp - lq)
        kl_qp += q * (lq - lp)
        tv += abs(p - q)
    return {"kl_pq": kl_pq, "kl_qp": kl_qp, "tv": tv / 2.0}


def top2(dense: list[float]) -> tuple[int, float]:
    """`(argmax, best - second best)` in one pass over a dense logprob list.

    The gap is how far this prompt is from flipping its argmax. It is
    reported, never gated: the point is that the top-1 *gate* stops being
    an assumption. Renormalization cancels in the subtraction, so the
    number is the same on raw logits, on logprobs, and on either side's
    un-renormalized dump — no need to agree on a normalizer first.

    Computed from the dense array rather than from the reference dump's
    first two rows, so it does not inherit the assumption that the dump
    arrived sorted.
    """
    best_i, best, second = 0, -math.inf, -math.inf
    for i, v in enumerate(dense):
        if v > best:
            best_i, second, best = i, best, v
        elif v > second:
            second = v
    return best_i, best - second


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rvmp", default="models/qwen3.rvmp",
                        help="installed .rvmp model dir")
    parser.add_argument("--ref", default="models/llamacpp-ref/llamacpp_ref",
                        help="reference dump dir (meta.json + single_*.npz)")
    parser.add_argument("--ramvamp",
                        help="ramvamp binary (default: cargo run --release)")
    parser.add_argument("--refresh", action="store_true",
                        help="recompute cached ramvamp-side dumps")
    parser.add_argument("--skip-longs", action="store_true",
                        help="skip the 512/1891/3492-token references. They "
                             "do not enter gate 3's mean, and recomputing "
                             "them (--refresh) costs ~2.5 h at phase-4 "
                             "speeds versus ~3 min for the short set")
    parser.add_argument("--timeout", type=float, default=10800.0,
                        help="per-prompt ramvamp timeout (s); the ~3500-token "
                        "long-context reference needs ~2h at phase-4 speeds")
    args = parser.parse_args()

    meta_path = os.path.join(args.ref, "meta.json")
    if not os.path.isfile(meta_path):
        fail(f"no meta.json under {args.ref}")
    with open(meta_path) as f:
        meta = json.load(f)
    vocab = meta["vocab"]

    results = []
    # Prompts that exist in the fixed gate set but could not be scored at an
    # identical context. They are NOT silently dropped: gate 3's mean is
    # defined over the whole fixed prompt set, so a shrunken denominator is
    # a gate failure, not a smaller gate.
    unscored = []
    for entry in meta["singles"]:
        name, prompt = entry["file"], entry["prompt"]
        if entry["depth"] != vocab:
            print(f"[{name}] SKIP: depth {entry['depth']} < vocab {vocab}",
                  file=sys.stderr)
            unscored.append({"file": name, "prompt": prompt,
                             "reason": f"reference depth {entry['depth']} "
                                       f"< vocab {vocab}",
                             "tokenizer_match": None})
            continue
        t0 = time.time()

        arrays = read_npz(os.path.join(args.ref, f"{name}.npz"))
        _, ids = arrays["token_ids"]
        _, lps = arrays["logprobs"]
        p_lp = [0.0] * vocab
        seen = [False] * vocab
        for tid, lp in zip(ids, lps):
            p_lp[tid] = lp
            seen[tid] = True
        if not all(seen):
            fail(f"{name}.npz does not cover the full vocab")

        rv = ramvamp_logits(args, prompt,
                            os.path.join(args.ref, f"rv_{name}.json"), vocab)
        q_lp = [0.0] * vocab
        seen = [False] * vocab
        for row in rv["top"]:
            q_lp[row["token_id"]] = row["logprob"]
            seen[row["token_id"]] = True
        if not all(seen):
            fail(f"ramvamp dump for {name} does not cover the full vocab")

        # Same-context check: both sides tokenized the prompt themselves.
        # A mismatch means the two distributions are conditioned on
        # different contexts, so their KL measures the tokenizer, not the
        # forward pass. The long-context branch below already skips on this;
        # the singles path used to warn and let the number into the mean.
        with gzip.open(os.path.join(args.ref, f"{name}.json.gz"), "rt") as f:
            raw = json.load(f)
        n_theirs = raw.get("tokens_evaluated")
        n_ours = rv.get("prompt_tokens")
        if n_theirs is None:
            print(f"[{name}] TOKENIZER UNVERIFIED: the reference dump has no "
                  f"`tokens_evaluated`, so the two contexts cannot be shown "
                  f"to match — KL skipped", file=sys.stderr)
            unscored.append({"file": name, "prompt": prompt,
                             "reason": "reference has no tokens_evaluated",
                             "tokenizer_match": None})
            continue
        if n_theirs != n_ours:
            print(f"[{name}] TOKENIZER MISMATCH: prompt token counts differ "
                  f"(llama.cpp {n_theirs}, ramvamp {n_ours}) — "
                  f"distributions are at different contexts, KL skipped",
                  file=sys.stderr)
            unscored.append({"file": name, "prompt": prompt,
                             "reason": f"llama.cpp {n_theirs} prompt tokens, "
                                       f"ramvamp {n_ours}",
                             "tokenizer_match": False})
            continue

        m = compare(p_lp, q_lp)
        top1_p = ids[0]
        top1_q = rv["top"][0]["token_id"]
        # Reported, not gated: how close each side came to flipping its own
        # argmax. See the module docstring — the top-1 condition is the one
        # part of gate 3 that had no recorded margin behind it.
        _, margin_p = top2(p_lp)
        _, margin_q = top2(q_lp)
        results.append({
            "file": name, "prompt": prompt,
            "kl_llama_vs_ramvamp": m["kl_pq"],
            "kl_ramvamp_vs_llama": m["kl_qp"],
            "total_variation": m["tv"],
            "top1_llamacpp": top1_p, "top1_ramvamp": top1_q,
            "top1_agree": top1_p == top1_q,
            "top1_margin_llamacpp": margin_p,
            "top1_margin_ramvamp": margin_q,
            "tokenizer_match": True,
        })
        print(f"[{name}] KL(P||Q) {m['kl_pq']:.3e}  KL(Q||P) {m['kl_qp']:.3e}  "
              f"TV {m['tv']:.3e}  top1 {'agree' if top1_p == top1_q else 'DISAGREE'} "
              f"(margin {min(margin_p, margin_q):.3f} nats)  "
              f"({time.time() - t0:.0f}s)  {prompt!r}")

    # Long-context references (RoPE/KV/attention at depth). Reported
    # separately: gate 3's mean is defined over the short fixed prompt set.
    long_results = []
    for entry in ([] if args.skip_longs else meta.get("longs", [])):
        name = entry["file"]
        if entry["depth"] != vocab:
            print(f"[{name}] SKIP: depth {entry['depth']} < vocab {vocab}",
                  file=sys.stderr)
            continue
        t0 = time.time()
        with open(os.path.join(args.ref, f"{name}.txt"), encoding="utf-8") as f:
            prompt = f.read()

        arrays = read_npz(os.path.join(args.ref, f"{name}.npz"))
        _, ids = arrays["token_ids"]
        _, lps = arrays["logprobs"]
        _, ref_prompt_ids = arrays["prompt_ids"]
        p_lp = [0.0] * vocab
        for tid, lp in zip(ids, lps):
            p_lp[tid] = lp

        rv = ramvamp_logits(args, prompt,
                            os.path.join(args.ref, f"rv_{name}.json"), vocab)
        if list(ref_prompt_ids) != list(rv["prompt_ids"]):
            theirs, ours = list(ref_prompt_ids), list(rv["prompt_ids"])
            first = next((i for i, (a, b) in enumerate(zip(theirs, ours))
                          if a != b), min(len(theirs), len(ours)))
            print(f"[{name}] TOKENIZER MISMATCH: llama.cpp {len(theirs)} ids, "
                  f"ramvamp {len(ours)} ids, first divergence at {first} — "
                  f"KL skipped (different contexts)", file=sys.stderr)
            long_results.append({"file": name, "tokens": entry["tokens"],
                                 "tokenizer_match": False})
            continue
        q_lp = [0.0] * vocab
        for row in rv["top"]:
            q_lp[row["token_id"]] = row["logprob"]

        m = compare(p_lp, q_lp)
        top1_p, top1_q = ids[0], rv["top"][0]["token_id"]
        _, margin_p = top2(p_lp)
        _, margin_q = top2(q_lp)
        long_results.append({
            "file": name, "tokens": entry["tokens"], "tokenizer_match": True,
            "kl_llama_vs_ramvamp": m["kl_pq"],
            "kl_ramvamp_vs_llama": m["kl_qp"],
            "total_variation": m["tv"],
            "top1_agree": top1_p == top1_q,
            "top1_margin_llamacpp": margin_p,
            "top1_margin_ramvamp": margin_q,
        })
        print(f"[{name}] ctx {entry['tokens']:>4} tok  KL(P||Q) {m['kl_pq']:.3e}  "
              f"KL(Q||P) {m['kl_qp']:.3e}  TV {m['tv']:.3e}  "
              f"top1 {'agree' if top1_p == top1_q else 'DISAGREE'} "
              f"(margin {min(margin_p, margin_q):.3f} nats)  "
              f"({time.time() - t0:.0f}s)")

    if not results:
        fail("no full-vocab reference entries to compare against")

    mean_pq = sum(r["kl_llama_vs_ramvamp"] for r in results) / len(results)
    mean_qp = sum(r["kl_ramvamp_vs_llama"] for r in results) / len(results)
    agree = sum(r["top1_agree"] for r in results)
    n_expected = len(meta["singles"])

    # Three independent conditions, all of which must hold. The mean alone
    # is not a gate: it is an average over 8 prompts and one blown prompt
    # hides inside it.
    problems = []
    if mean_pq > KL_TARGET:
        problems.append(f"mean KL {mean_pq:.3e} > {KL_TARGET:g}")
    for r in results:
        if r["kl_llama_vs_ramvamp"] > KL_PROMPT_CEILING:
            problems.append(
                f"{r['file']}: KL {r['kl_llama_vs_ramvamp']:.3e} > "
                f"per-prompt ceiling {KL_PROMPT_CEILING:g}")
    for r in results:
        if not r["top1_agree"]:
            problems.append(
                f"{r['file']}: top-1 disagrees (llama.cpp "
                f"{r['top1_llamacpp']}, ramvamp {r['top1_ramvamp']})")
    for r in unscored:
        problems.append(
            f"{r['file']}: not scored ({r['reason']}) — gate 3's mean is "
            f"defined over all {n_expected} fixed prompts")

    # Margin behind the top-1 condition, reported so the gate's most
    # flake-prone condition is measured rather than assumed.
    tightest = min(results,
                   key=lambda r: min(r["top1_margin_llamacpp"],
                                     r["top1_margin_ramvamp"]))
    min_margin_p = min(r["top1_margin_llamacpp"] for r in results)
    min_margin_q = min(r["top1_margin_ramvamp"] for r in results)

    verdict = "PASS" if not problems else "FAIL"
    print(f"\nprompts scored: {len(results)}/{n_expected}  "
          f"top-1 agreement: {agree}/{len(results)}")
    print(f"mean KL(llama.cpp || ramvamp): {mean_pq:.3e}")
    print(f"mean KL(ramvamp || llama.cpp): {mean_qp:.3e}")
    print(f"worst single prompt KL(P||Q):  "
          f"{max(r['kl_llama_vs_ramvamp'] for r in results):.3e}")
    print(f"tightest top-1 margin (nats):  llama.cpp {min_margin_p:.3f}, "
          f"ramvamp {min_margin_q:.3f}  (each a min over prompts; nearest "
          f"tie overall {tightest['file']}; reported, not gated)")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print(f"gate 3 (mean KL <= {KL_TARGET:g}, every prompt <= "
          f"{KL_PROMPT_CEILING:g}, top-1 {len(results)}/{len(results)}, all "
          f"{n_expected} prompts scored): {verdict}")

    summary = {
        "reference": {
            "llama_server_version": meta.get("llama_server_version"),
            "gguf": meta.get("gguf"),
            "captured": meta.get("date"),
        },
        "vocab": vocab,
        "kl_target": KL_TARGET,
        "kl_prompt_ceiling": KL_PROMPT_CEILING,
        "mean_kl_llama_vs_ramvamp": mean_pq,
        "mean_kl_ramvamp_vs_llama": mean_qp,
        "max_kl_llama_vs_ramvamp": max(r["kl_llama_vs_ramvamp"] for r in results),
        "top1_agreement": f"{agree}/{len(results)}",
        # Not a gate input. The margin behind the top-1 condition, so a
        # future top-1 FAIL can be read as "the arithmetic moved" or "this
        # prompt was always a near-tie" instead of guessed at.
        "min_top1_margin_llamacpp": min_margin_p,
        "min_top1_margin_ramvamp": min_margin_q,
        "tightest_top1_margin_file": tightest["file"],
        "prompts_scored": f"{len(results)}/{n_expected}",
        "problems": problems,
        "verdict": verdict,
        "per_prompt": results,
        "unscored": unscored,
        "long_context": long_results,
    }
    out_path = os.path.join(args.ref, "kl_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
