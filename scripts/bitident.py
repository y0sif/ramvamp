#!/usr/bin/env python3
"""Bit-identity fingerprint of the forward pass: capture once, compare later.

The primary phase-5 regression gate. Phase 5 replaces synchronous expert
preads with io_uring + O_DIRECT + a per-layer cache + pinned compute
threads, and is *designed* to change only when and by whom bytes are
fetched, never the arithmetic or its order. So the strongest available
gate is not a tolerance: it is "the logits are byte-identical to phase 4".

Anything that perturbs accumulation order — a different thread doing a
reduction, a partial tile summed in a new order, a cache handing back a
stale-but-plausible expert — moves logits in the last mantissa bits and
shows up here immediately, while a KL gate at 3e-2 (which sits *above*
the 0.5-1.3e-2 intra-engine float-reordering noise floor, EXP-004) would
not notice.

Usage:

    scripts/bitident.py capture models/llamacpp-ref/phase4-baseline
    scripts/bitident.py compare models/llamacpp-ref/phase4-baseline

`capture` runs `ramvamp logits --top N --skip-hashes` over the 8 fixed
single prompts from the reference `meta.json`, and writes each raw stdout
plus a `manifest.json` of per-prompt SHA-256 digests, the binary's
identity (path, size, mtime, sha256), the git commit, and the model dir.

`compare` re-runs the current build over the same prompts and diffs.
Identical bytes -> PASS, exit 0. Differing bytes -> FAIL, exit 1, with,
per prompt: whether the difference is numeric or formatting-only, the
first differing entry of the `top` array, the token ids at that rank on
both sides, and the magnitude of the logit delta in absolute terms and in
ULPs. That distinction matters: a reordered JSON key or a new field is a
harness artifact; a 1-ULP logit move at rank 3000 is a real regression.

`--new DIR` compares an already-captured directory instead of re-running,
so two baselines taken at different commits can be diffed offline. Both
sides are re-hashed from the files on disk in either mode — the recorded
digests are only cross-checked against them, never substituted for them,
because a capture directory can be regenerated without its manifest.

An empty manifest is a failed capture and is rejected, not reported as
`PASS 0/0`.

Baseline captured 2026-08-03 on d80cc84 + scripts (phase-4 code, release
build, 185H): 8/8 prompts, --top 4096, and an immediate second run
compared byte-clean — the fingerprint is reproducible run to run, so a
future mismatch is a code change and not harness noise.

Exit codes, shared with the repo's other gate scripts (`cold_bench.py`,
`greedy_regression.py`, `kl_vs_reference.py`, `lfu_sim.py`):

  0  the gate ran and passed (or `capture` wrote a baseline)
  1  the gate ran and failed — bytes differ, the forward pass changed
  2  the gate could not run: the directory is not a capture, its manifest
     lists no prompts, a listed payload is missing, a payload and its
     manifest disagree, the capture directory exists and is not empty
     without --force, ramvamp failed to start or timed out

Python stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time

# Rank depth of the fingerprint. The sort in `ramvamp logits` is over the
# whole 151936-token vocabulary regardless, so a deeper --top costs only
# JSON; 4096 reaches well into the low-probability tail, where a change in
# accumulation order shows up first (top-1 is usually robust to it).
DEFAULT_TOP = 4096

MANIFEST = "manifest.json"


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        rev = out.stdout.strip() if out.returncode == 0 else "unknown"
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            rev += "-dirty"
        return rev
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def binary_identity(args) -> dict:
    """Identify what produced the numbers, so a stale capture is obvious."""
    if not args.ramvamp:
        return {"invocation": "cargo run --release -p ramvamp"}
    path = os.path.abspath(args.ramvamp)
    if not os.path.isfile(path):
        fail(f"no ramvamp binary at {path}")
    st = os.stat(path)
    return {
        "invocation": path,
        "bytes": st.st_size,
        "mtime": st.st_mtime,
        "sha256": sha256_file(path),
    }


def prompts_from_meta(ref: str) -> list[tuple[str, str]]:
    meta_path = os.path.join(ref, "meta.json")
    if not os.path.isfile(meta_path):
        fail(f"no meta.json under {ref}")
    with open(meta_path) as f:
        meta = json.load(f)
    entries = [(e["file"], e["prompt"]) for e in meta["singles"]]
    if not entries:
        fail(f"{meta_path} lists no singles")
    return entries


def run_logits(args, prompt: str) -> str:
    cmd = [args.ramvamp] if args.ramvamp else [
        "cargo", "run", "--release", "--quiet", "-p", "ramvamp", "--"
    ]
    cmd += [
        "logits",
        "--model", args.rvmp,
        "--prompt", prompt,
        "--top", str(args.top),
        "--skip-hashes",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, check=False
        )
    except subprocess.TimeoutExpired:
        fail(f"ramvamp timed out after {args.timeout:.0f}s on {prompt[:60]!r}")
    if result.returncode != 0:
        fail(f"ramvamp exited {result.returncode} on {prompt[:60]!r}\n"
             f"stderr:\n{result.stderr[-2000:]}")
    return result.stdout


def capture(args, out_dir: str, quiet: bool = False) -> dict:
    entries = prompts_from_meta(args.ref)
    os.makedirs(out_dir, exist_ok=True)
    records = []
    t_start = time.time()
    for name, prompt in entries:
        t0 = time.time()
        stdout = run_logits(args, prompt)
        blob = stdout.encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        with open(os.path.join(out_dir, f"{name}.json"), "wb") as f:
            f.write(blob)
        records.append({
            "file": name, "prompt": prompt,
            "bytes": len(blob), "sha256": digest,
        })
        if not quiet:
            print(f"[{name}] {digest[:16]}...  {len(blob):>9} B  "
                  f"({time.time() - t0:.0f}s)  {prompt!r}")
    manifest = {
        "tool": "scripts/bitident.py",
        "captured": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_commit": git_commit(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "model": os.path.abspath(args.rvmp),
        "reference": os.path.abspath(args.ref),
        "top": args.top,
        "binary": binary_identity(args),
        "elapsed_s": round(time.time() - t_start, 1),
        "prompts": records,
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_manifest(d: str) -> dict:
    path = os.path.join(d, MANIFEST)
    if not os.path.isfile(path):
        fail(f"{d} is not a bitident capture (no {MANIFEST})")
    with open(path) as f:
        manifest = json.load(f)
    if not manifest.get("prompts"):
        fail(f"{path} lists no prompts; there is nothing to fingerprint. "
             f"A capture that recorded 0 prompts is a failed capture, not a "
             f"passing comparison — re-run `bitident.py capture`.")
    return manifest


def actual_digests(d: str, manifest: dict, what: str) -> dict[str, str]:
    """Re-hash the capture's files. Never trust the recorded digest.

    The manifest is just a file next to the payloads: a directory whose
    `single_*.json` were regenerated but whose `manifest.json` was not would
    otherwise compare as identical. Digests are cheap (8 x ~0.5 MB), the
    whole point of the tool is byte-identity, so they are recomputed on
    every side of every comparison and cross-checked against what the
    manifest claims.
    """
    out: dict[str, str] = {}
    for rec in manifest["prompts"]:
        name = rec["file"]
        path = os.path.join(d, f"{name}.json")
        if not os.path.isfile(path):
            fail(f"{what}: {path} is listed in {MANIFEST} but does not exist")
        digest = sha256_file(path)
        if digest != rec.get("sha256"):
            fail(f"{what}: {path} hashes to {digest[:16]}... but {MANIFEST} "
                 f"records {str(rec.get('sha256'))[:16]}... — the capture "
                 f"directory and its manifest disagree, so neither can be "
                 f"used as evidence. Re-capture it.")
        out[name] = digest
    return out


def ulps(a: float, b: float) -> int:
    """Distance between two f32 values in representable steps."""
    def key(x: float) -> int:
        (bits,) = struct.unpack("<I", struct.pack("<f", x))
        return bits if bits < 0x8000_0000 else 0x1_0000_0000 - bits
    return abs(key(a) - key(b))


def diff_prompt(old_path: str, new_path: str) -> dict:
    """Locate the first divergence and say whether it is numeric."""
    with open(old_path, encoding="utf-8") as f:
        old_raw = f.read()
    with open(new_path, encoding="utf-8") as f:
        new_raw = f.read()
    if old_raw == new_raw:
        return {"kind": "identical"}

    try:
        old = json.loads(old_raw)
        new = json.loads(new_raw)
    except json.JSONDecodeError as e:
        return {"kind": "unparseable", "detail": str(e)}

    if old.get("prompt_ids") != new.get("prompt_ids"):
        return {
            "kind": "tokenizer",
            "detail": f"prompt_ids differ: {len(old.get('prompt_ids') or [])} vs "
                      f"{len(new.get('prompt_ids') or [])} ids",
        }

    o_top, n_top = old.get("top", []), new.get("top", [])
    if len(o_top) != len(n_top):
        return {"kind": "depth",
                "detail": f"top depth {len(o_top)} vs {len(n_top)}"}

    for rank, (a, b) in enumerate(zip(o_top, n_top)):
        if a.get("token_id") != b.get("token_id") or a.get("logit") != b.get("logit"):
            la, lb = float(a.get("logit", 0.0)), float(b.get("logit", 0.0))
            return {
                "kind": "numeric",
                "rank": rank,
                "old_token": a.get("token_id"),
                "new_token": b.get("token_id"),
                "old_logit": la,
                "new_logit": lb,
                "abs_delta": abs(lb - la),
                "ulps": ulps(la, lb),
                "reordered": a.get("token_id") != b.get("token_id"),
            }
    return {"kind": "formatting"}


def compare(args, base_dir: str) -> int:
    base = load_manifest(base_dir)
    if base["top"] != args.top:
        print(f"note: baseline used --top {base['top']}, overriding "
              f"--top {args.top} to match", file=sys.stderr)
        args.top = base["top"]

    if args.new:
        new_dir, new_manifest, temp = args.new, load_manifest(args.new), False
        print(f"comparing {base_dir} vs {new_dir} (no run)")
    else:
        new_dir = args.workdir or os.path.join(
            base_dir + ".compare", time.strftime("%Y%m%dT%H%M%S"))
        temp = args.workdir is None
        print(f"re-running current build into {new_dir}")
        new_manifest = capture(args, new_dir)

    print(f"baseline: {base['git_commit']}  captured {base['captured']}  "
          f"top {base['top']}")
    print(f"current : {new_manifest['git_commit']}  "
          f"captured {new_manifest['captured']}")
    if base["binary"].get("sha256") and new_manifest["binary"].get("sha256"):
        same = base["binary"]["sha256"] == new_manifest["binary"]["sha256"]
        print(f"binary  : {'identical' if same else 'DIFFERENT'} "
              f"({base['binary']['sha256'][:12]} -> "
              f"{new_manifest['binary']['sha256'][:12]})")
    if base["model"] != new_manifest["model"]:
        print(f"WARNING: model dir changed ({base['model']} -> "
              f"{new_manifest['model']})", file=sys.stderr)

    # Both sides are re-hashed from the files on disk; the manifests only
    # say which files to look at. In `--new DIR` mode nothing was run here,
    # so the recorded digests are the *only* thing a naive comparison would
    # look at, and a directory holding changed payloads under a stale
    # manifest would report identical.
    base_digests = actual_digests(base_dir, base, f"baseline {base_dir}")
    new_digests = actual_digests(new_dir, new_manifest, f"comparand {new_dir}")

    new_by_file = {r["file"]: r for r in new_manifest["prompts"]}
    differing = []
    for rec in base["prompts"]:
        name = rec["file"]
        got = new_by_file.get(name)
        if got is None:
            print(f"[{name}] MISSING in the new capture")
            differing.append((name, {"kind": "missing"}))
            continue
        if new_digests[name] == base_digests[name]:
            print(f"[{name}] identical  {base_digests[name][:16]}...")
            continue
        detail = diff_prompt(os.path.join(base_dir, f"{name}.json"),
                             os.path.join(new_dir, f"{name}.json"))
        differing.append((name, detail))
        if detail["kind"] == "numeric":
            print(f"[{name}] DIFFERS (numeric): first at rank {detail['rank']} "
                  f"token {detail['old_token']} -> {detail['new_token']}, "
                  f"logit {detail['old_logit']!r} -> {detail['new_logit']!r} "
                  f"(|d| {detail['abs_delta']:.3e}, {detail['ulps']} ULP"
                  f"{'s' if detail['ulps'] != 1 else ''}"
                  f"{', rank reordered' if detail['reordered'] else ''})")
        elif detail["kind"] == "formatting":
            print(f"[{name}] DIFFERS (formatting only): every token_id and "
                  f"logit matches; the JSON envelope changed")
        else:
            print(f"[{name}] DIFFERS ({detail['kind']}): "
                  f"{detail.get('detail', '')}")

    if temp and not args.keep:
        shutil.rmtree(os.path.dirname(new_dir), ignore_errors=True)

    n = len(base["prompts"])
    if not differing:
        print(f"\nbit-identity: PASS  {n}/{n} prompts byte-identical to "
              f"{base_dir}")
        return 0
    numeric = [d for _, d in differing if d["kind"] == "numeric"]
    kinds = ", ".join(sorted({d["kind"] for _, d in differing}))
    print(f"\nbit-identity: FAIL  {len(differing)}/{n} prompts differ "
          f"({kinds}); {len(numeric)} numeric -> the forward pass changed")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("dir", help="baseline directory to write or compare to")
    parser.add_argument("--rvmp", default="models/qwen3.rvmp",
                        help="installed .rvmp model dir")
    parser.add_argument("--ref", default="models/llamacpp-ref/llamacpp_ref",
                        help="reference dir supplying the prompt set (meta.json)")
    parser.add_argument("--ramvamp",
                        help="ramvamp binary (default: cargo run --release)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"fingerprint depth (default {DEFAULT_TOP}); "
                             "compare mode adopts the baseline's value")
    parser.add_argument("--new", help="compare mode: diff this existing "
                                      "capture instead of re-running")
    parser.add_argument("--workdir", help="compare mode: keep the fresh "
                                          "capture here instead of a temp dir")
    parser.add_argument("--keep", action="store_true",
                        help="compare mode: do not delete the temp capture")
    parser.add_argument("--force", action="store_true",
                        help="capture mode: overwrite a non-empty directory")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="per-prompt ramvamp timeout (s)")
    args = parser.parse_args()

    if args.mode == "capture":
        if os.path.isdir(args.dir) and os.listdir(args.dir) and not args.force:
            fail(f"{args.dir} exists and is not empty (use --force)")
        manifest = capture(args, args.dir)
        print(f"\ncaptured {len(manifest['prompts'])} prompts at --top "
              f"{manifest['top']} in {manifest['elapsed_s']:.0f}s -> {args.dir}")
        print(f"commit {manifest['git_commit']}")
        return 0
    return compare(args, args.dir)


if __name__ == "__main__":
    sys.exit(main())
