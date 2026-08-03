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

Python stdlib only: the .npz reader below covers exactly what
numpy.savez_compressed writes (v1/v2 .npy headers, little-endian scalar
dtypes, C order).

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
# AVX2 on 2 of 3 prompts. 3e-2 sits ~3x above the measured mean and ~2x
# above the worst single prompt; perplexity (gate 5) is the quality
# backstop.
KL_TARGET = 3e-2


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
    for entry in meta["singles"]:
        name, prompt = entry["file"], entry["prompt"]
        if entry["depth"] != vocab:
            print(f"[{name}] SKIP: depth {entry['depth']} < vocab {vocab}",
                  file=sys.stderr)
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
        with gzip.open(os.path.join(args.ref, f"{name}.json.gz"), "rt") as f:
            raw = json.load(f)
        n_theirs = raw.get("tokens_evaluated")
        n_ours = rv.get("prompt_tokens")
        if n_theirs is not None and n_theirs != n_ours:
            print(f"[{name}] WARNING: prompt token counts differ "
                  f"(llama.cpp {n_theirs}, ramvamp {n_ours}) — "
                  f"distributions are at different contexts", file=sys.stderr)

        m = compare(p_lp, q_lp)
        top1_p = ids[0]
        top1_q = rv["top"][0]["token_id"]
        results.append({
            "file": name, "prompt": prompt,
            "kl_llama_vs_ramvamp": m["kl_pq"],
            "kl_ramvamp_vs_llama": m["kl_qp"],
            "total_variation": m["tv"],
            "top1_llamacpp": top1_p, "top1_ramvamp": top1_q,
            "top1_agree": top1_p == top1_q,
        })
        print(f"[{name}] KL(P||Q) {m['kl_pq']:.3e}  KL(Q||P) {m['kl_qp']:.3e}  "
              f"TV {m['tv']:.3e}  top1 {'agree' if top1_p == top1_q else 'DISAGREE'} "
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
        long_results.append({
            "file": name, "tokens": entry["tokens"], "tokenizer_match": True,
            "kl_llama_vs_ramvamp": m["kl_pq"],
            "kl_ramvamp_vs_llama": m["kl_qp"],
            "total_variation": m["tv"],
            "top1_agree": top1_p == top1_q,
        })
        print(f"[{name}] ctx {entry['tokens']:>4} tok  KL(P||Q) {m['kl_pq']:.3e}  "
              f"KL(Q||P) {m['kl_qp']:.3e}  TV {m['tv']:.3e}  "
              f"top1 {'agree' if top1_p == top1_q else 'DISAGREE'} "
              f"({time.time() - t0:.0f}s)")

    if not results:
        fail("no full-vocab reference entries to compare against")

    mean_pq = sum(r["kl_llama_vs_ramvamp"] for r in results) / len(results)
    mean_qp = sum(r["kl_ramvamp_vs_llama"] for r in results) / len(results)
    agree = sum(r["top1_agree"] for r in results)
    verdict = "PASS" if mean_pq <= KL_TARGET else "FAIL"
    print(f"\nprompts: {len(results)}  top-1 agreement: {agree}/{len(results)}")
    print(f"mean KL(llama.cpp || ramvamp): {mean_pq:.3e}")
    print(f"mean KL(ramvamp || llama.cpp): {mean_qp:.3e}")
    print(f"gate 3 (mean KL <= {KL_TARGET:g}): {verdict}")

    summary = {
        "reference": {
            "llama_server_version": meta.get("llama_server_version"),
            "gguf": meta.get("gguf"),
            "captured": meta.get("date"),
        },
        "vocab": vocab,
        "kl_target": KL_TARGET,
        "mean_kl_llama_vs_ramvamp": mean_pq,
        "mean_kl_ramvamp_vs_llama": mean_qp,
        "top1_agreement": f"{agree}/{len(results)}",
        "verdict": verdict,
        "per_prompt": results,
        "long_context": long_results,
    }
    out_path = os.path.join(args.ref, "kl_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
