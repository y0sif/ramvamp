#!/usr/bin/env python3
"""Gate 5: WikiText-2 perplexity, ramvamp against llama-perplexity, window by window.

Closes validation gate 5, the last open v0 gate (docs/roadmap.md, "Gate 5:
perplexity"). The reference side is banked: llama.cpp b10217's

    llama-perplexity -m <Q4_K_M GGUF> -f wiki.test.raw -c 512 --chunks 40

on the same GGUF bytes the runtime installs from, whose log carries the
running perplexity after every window (`[1]4.8717,...,[40]6.3810`) and the
final `PPL = 6.3810 +/- 0.16588`. This script runs `ramvamp perplexity` over
the same corpus at the same window and count and compares the two logs.

The protocol, read from b10217's `tools/perplexity/perplexity.cpp` and
`common/arg.cpp` rather than assumed:

  * `-f` reads the file and drops one trailing newline; escape sequences
    are then processed (a no-op here: this script refuses a corpus with a
    backslash, because ramvamp does not repeat that step).
  * The text is tokenized once, whole, with `add_special` set, which adds a
    BOS only when the model asks for one. The Qwen3 GGUF sets
    `tokenizer.ggml.add_bos_token = false`, so no BOS is prepended and no
    window's first token is overwritten with one.
  * Window k is tokens [512k, 512k + 512), run from an empty KV cache.
  * Positions 256..510 of a window are scored against the token after each:
    255 tokens a window, 10,200 over 40 windows.
  * log-softmax: the max in f32, exp(l - max) in f32 summed in f64.
    `--batch 2048` packs four windows per decode (n_seq = 4), which does not
    change any window's arithmetic.

Before comparing, the token streams are checked: llama-tokenize (b10217, the
same `common_tokenize` call on the same text) must produce exactly the ids
ramvamp scored, or the windows are not the same windows and the comparison
means nothing. A mismatch fails the gate, as a tokenizer mismatch fails
gate 3.

The pass criterion, set by the author on 2026-10-12 before the ramvamp side
had ever run:

  * the final PPL within 1% of the reference's, and
  * the running PPL after every window within 1% of the reference's running
    value after the same window.

The reference's +/-0.16588 (2.6%) is not the tolerance. It is the sampling
error of one model's perplexity over this corpus, and both engines score
the identical 10,200 tokens, so it would let through a gap several times
larger than float reordering produces (gate 3 measured that noise as a
full-vocab KL of 4.5e-3 to 1.3e-2). A protocol slip, such as the wrong
scored positions or a BOS, moves perplexity by several percent and shows up
in the running value from the first windows on, which is why every window
is gated and not only the last.

Alongside the gate, and not part of it, the per-window mean NLL is recovered
from both running series (the reference prints four decimals, so its
per-window figures carry about 1e-3 nats of rounding) and the paired
difference is summarized.

Exit status: 0 PASS, 1 FAIL, 2 error (bad inputs, a tool that failed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys

# Set by the author on 2026-10-12, before the ramvamp side was run.
TOLERANCE = 0.01

DEFAULT_RVMP = "models/qwen3.rvmp"
DEFAULT_GGUF = "models/gguf/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
DEFAULT_CORPUS = "scratch/gatefix-demo/klref/wiki.test.raw"
DEFAULT_REFERENCE = "models/llamacpp-ref/llamacpp_ref/ppl_wikitext_c512x40.txt"
# WikiText-2 raw test split, as the reference run read it.
CORPUS_MD5 = "7c0137fc034ddbc56a296bce31b4f7fb"


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def parse_reference(path: str) -> dict:
    """The running series, the final estimate and the run's shape."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        fail(f"reading the reference log: {e}")
    pairs = re.findall(r"\[(\d+)\]([0-9]+\.[0-9]+),", text)
    final = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text)
    shape = re.search(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", text)
    if not pairs or not final or not shape:
        fail(f"{path} is not a llama-perplexity log with running values")
    indices = [int(k) for k, _ in pairs]
    if indices != list(range(1, len(indices) + 1)):
        fail(f"{path}: running values are not windows 1..{len(indices)} in order")
    chunks, n_ctx = int(shape.group(1)), int(shape.group(2))
    if len(indices) != chunks:
        fail(f"{path}: {len(indices)} running values for {chunks} chunks")
    return {
        "running": [float(v) for _, v in pairs],
        "final": float(final.group(1)),
        "error": float(final.group(2)),
        "chunks": chunks,
        "n_ctx": n_ctx,
    }


def window_nll(running: list[float], per_window: int) -> list[float]:
    """Mean NLL of each window, from a running-perplexity series."""
    totals = [k * per_window * math.log(p) for k, p in enumerate(running, start=1)]
    return [(t - (totals[i - 1] if i else 0.0)) / per_window for i, t in enumerate(totals)]


def run(cmd: list[str], what: str, timeout: float) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        fail(f"{what}: {e}")
    if result.returncode != 0:
        fail(f"{what} exited {result.returncode}\nstderr:\n{result.stderr[-2000:]}")
    return result


def llama_ids(args, text: str) -> list[int]:
    """llama-tokenize over the text llama-perplexity scored."""
    path = os.path.join(args.workdir, "corpus-as-scored.txt")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    result = run(
        [args.llama_tokenize, "-m", args.gguf, "-f", path, "--ids", "--log-disable"],
        "llama-tokenize",
        timeout=600,
    )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        fail(f"llama-tokenize printed no id list:\n{result.stdout[-500:]}")


def ramvamp_scores(args, chunks: int, n_ctx: int) -> dict:
    """`ramvamp perplexity`'s JSON, run now or reused from --reuse."""
    if args.reuse:
        with open(args.reuse) as f:
            return json.load(f)
    out = os.path.join(args.workdir, "ramvamp_perplexity.json")
    cmd = [
        args.ramvamp, "perplexity",
        "--model", args.rvmp,
        "--file", args.corpus,
        "--window", str(n_ctx),
        "--chunks", str(chunks),
        "--json", out,
        "--skip-hashes",
    ] + args.ramvamp_arg
    result = run(cmd, "ramvamp perplexity", timeout=args.timeout)
    with open(os.path.join(args.workdir, "ramvamp_perplexity.log"), "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    with open(out) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ramvamp", default="target/release/ramvamp")
    parser.add_argument("--rvmp", default=DEFAULT_RVMP)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--llama-tokenize", default="llama-tokenize")
    parser.add_argument("--workdir", required=True,
                        help="where the ramvamp JSON, its log and the verdict land")
    parser.add_argument("--reuse", metavar="JSON",
                        help="score a saved `ramvamp perplexity --json` instead of running it")
    parser.add_argument("--ramvamp-arg", action="append", default=[],
                        help="extra ramvamp flag, repeatable (`--ramvamp-arg=--flag`)")
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    args = parser.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    raw = open(args.corpus, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != CORPUS_MD5:
        fail(f"{args.corpus} has md5 {md5}, not the reference corpus's {CORPUS_MD5}")
    if b"\\" in raw:
        fail("the corpus has a backslash: llama's escape processing would rewrite it "
             "and ramvamp does not, so the two would not score the same text")
    text = raw.decode("utf-8")
    scored_text = text[:-1] if text.endswith("\n") else text

    ref = parse_reference(args.reference)
    chunks, n_ctx = ref["chunks"], ref["n_ctx"]
    per_window = n_ctx - 1 - n_ctx // 2

    rv = ramvamp_scores(args, chunks, n_ctx)
    if (rv["window"], rv["windows"], rv["first_scored"], rv["scored_per_window"]) != (
        n_ctx, chunks, n_ctx // 2, per_window
    ):
        fail(f"ramvamp ran window {rv['window']} x {rv['windows']} scoring "
             f"{rv['scored_per_window']} from {rv['first_scored']}; the reference is "
             f"{n_ctx} x {chunks} scoring {per_window} from {n_ctx // 2}")

    problems: list[str] = []

    lids = llama_ids(args, scored_text)
    used = chunks * n_ctx
    tokens_match = lids[:used] == rv["ids"]
    if not tokens_match:
        at = next((i for i, (a, b) in enumerate(zip(lids, rv["ids"])) if a != b),
                  min(len(lids), len(rv["ids"])))
        problems.append(f"token streams differ at id {at} (window {at // n_ctx + 1}): "
                        f"llama {lids[at:at + 5]} vs ramvamp {rv['ids'][at:at + 5]}")
    if len(lids) != rv["tokens"]:
        problems.append(f"the corpus tokenizes to {len(lids)} ids in llama.cpp and "
                        f"{rv['tokens']} in ramvamp")

    rv_running = [w["running_ppl"] for w in rv["per_window"]]
    worst = (0.0, 0)
    rows = []
    for k, (a, b) in enumerate(zip(ref["running"], rv_running), start=1):
        rel = b / a - 1.0
        rows.append((k, a, b, rel))
        if abs(rel) > abs(worst[0]):
            worst = (rel, k)
        if abs(rel) > TOLERANCE:
            problems.append(f"window {k}: running PPL {b:.4f} against {a:.4f} "
                            f"({rel:+.3%}, over {TOLERANCE:.0%})")
    final_rel = rv["ppl"] / ref["final"] - 1.0
    if abs(final_rel) > TOLERANCE:
        problems.append(f"final PPL {rv['ppl']:.4f} against {ref['final']:.4f} "
                        f"({final_rel:+.3%}, over {TOLERANCE:.0%})")

    # Diagnostics, not gated.
    ref_nll = window_nll(ref["running"], per_window)
    rv_nll = [w["nll"] / w["count"] for w in rv["per_window"]]
    diffs = [b - a for a, b in zip(ref_nll, rv_nll)]
    mean_diff = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1))
    se = sd / math.sqrt(len(diffs))

    verdict = "PASS" if not problems else "FAIL"
    print(f"{'window':>6}  {'llama.cpp':>9}  {'ramvamp':>9}  {'rel':>8}")
    for k, a, b, rel in rows:
        print(f"{k:>6}  {a:>9.4f}  {b:>9.4f}  {rel:>+8.3%}")
    print(f"final   {ref['final']:.4f} +/- {ref['error']:.5f} (llama.cpp)  "
          f"{rv['ppl']:.4f} +/- {rv['ppl_error']:.5f} (ramvamp)  {final_rel:+.3%}")
    print(f"worst running gap {worst[0]:+.3%} at window {worst[1]}")
    print(f"per-window mean NLL, ramvamp - llama.cpp: {mean_diff:+.5f} nats "
          f"(sd {sd:.5f}, se {se:.5f}, n {len(diffs)}; reported, not gated)")
    print(f"token streams: {'identical' if tokens_match else 'DIFFERENT'} "
          f"over the {used} ids scored ({len(lids)} in the corpus)")
    for problem in problems:
        print(f"  FAILED: {problem}")
    print(f"gate 5 (final and every running PPL within {TOLERANCE:.0%} of llama.cpp "
          f"b10217, identical tokens): {verdict}")

    summary = {
        "reference": {"log": args.reference, "final": ref["final"], "error": ref["error"],
                      "chunks": chunks, "n_ctx": n_ctx, "llama_cpp": "b10217"},
        "corpus": {"path": args.corpus, "md5": md5, "bytes": len(raw)},
        "tolerance": TOLERANCE,
        "ramvamp": {"ppl": rv["ppl"], "ppl_error": rv["ppl_error"],
                    "prefill_mode": rv.get("prefill_mode"), "seconds": rv.get("seconds")},
        "final_rel": final_rel,
        "worst_running_rel": worst[0],
        "worst_running_window": worst[1],
        "tokens_identical": tokens_match,
        "tokens_llama": len(lids),
        "tokens_ramvamp": rv["tokens"],
        "per_window": [{"window": k, "llama": a, "ramvamp": b, "rel": rel,
                        "nll_llama": ref_nll[k - 1], "nll_ramvamp": rv_nll[k - 1]}
                       for k, a, b, rel in rows],
        "mean_nll_diff": mean_diff,
        "mean_nll_diff_se": se,
        "problems": problems,
        "verdict": verdict,
    }
    out = os.path.join(args.workdir, "gate5.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
