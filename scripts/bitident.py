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
    scripts/bitident.py capture <dir> --set longs      # chunk-boundary set
    scripts/bitident.py compare <dir> --dry-run        # list, run nothing

Every `models/...` path here is a default layout, not a promise that the
bytes are present. That tree is **not in the repository**: `.gitignore`
excludes /models/, so a fresh clone holds neither the reference dump nor
any captured baseline. The reference dump has to be re-banked from
llama.cpp b10217, and the baseline directory is written by this script's
own `capture` before any `compare` has something to compare against.

`capture` runs `ramvamp logits --top N --skip-hashes` over a prompt set
drawn from the reference `meta.json`, and writes each raw stdout plus a
`manifest.json` of per-prompt SHA-256 digests, the binary's identity
(path, size, mtime, sha256), the git commit, the model dir, and the
prompt set with each prompt's token count.

`--set` picks which prompts are fingerprinted:

  singles  (default)  the 8 short `meta["singles"]` prompts, 4-12 tokens
  longs               the 3 `meta["longs"]` fixtures, whose text lives in
                      `<ref>/long_NN.txt`: 512, 1891 and 3492 tokens
  all                 both, 11 prompts

The singles are a few tokens each, so they exercise one prefill chunk and
nothing else. The longs exist to pin the *chunked* prefill path: at a
512-token chunk size, `long_00` is exactly one full chunk with an empty
remainder (the boundary case that off-by-one errors land on), `long_01`
is 3 full chunks + 379, `long_02` is 6 full + 420. A prefill rewrite that
is correct on 5 tokens and wrong at a chunk seam passes `--set singles`
and fails `--set longs`, which is the entire point of capturing both.

Prompt sets are part of a baseline's identity, not a runtime option:
`compare` adopts the set recorded in the baseline's manifest exactly as
it adopts `--top`, so a baseline captured over singles never silently
starts including longs. Passing a `--set` that contradicts the baseline
is an error (exit 2), not a comparison over the intersection. Manifests
written before `--set` existed are read as `singles`.

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
`PASS 0/0`. So is a comparand whose prompt set is not the baseline's:
comparing 8 matching singles while quietly ignoring 3 unexamined longs
would report PASS over a set nobody chose.

`--dry-run` resolves the prompt set, prints what would be fingerprinted
(name, kind, token count, per-prompt timeout) and exits 0 without loading
the model — the cheap way to confirm which prompts a given invocation
actually selects.

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
     manifest disagree, the two sides fingerprint different prompt sets,
     the capture directory exists and is not empty without --force,
     ramvamp failed to start or timed out

A timeout is always exit 2. A prompt the runtime never finished produced
no bytes to compare, so it is "could not run", never "no difference".

Python stdlib only (argparse, hashlib, json, os, shutil, struct,
subprocess, sys, time).
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

# Which prompts make up the fingerprint. `singles` is the default and is
# the historical set: every baseline captured before this flag existed is
# a singles baseline, and `load_manifest` reads a missing `prompt_set` as
# such, so the phase-4 gate keeps comparing 8 prompts against 8 prompts.
PROMPT_SETS = ("singles", "longs", "all")
DEFAULT_PROMPT_SET = "singles"

# Per-prompt timeout. A flat 3600 s is generous for a 12-token single and
# marginal for a 3492-token long: phase-5 prefill runs ~1.4 tok/s, so
# long_02 alone is ~42 minutes of forward pass before the model load and
# the 151936-way sort. Rather than pick one number that is either useless
# or unbounded, the default scales with the prompt's known token count and
# floors at the old value; --timeout overrides it with a flat number for
# every prompt. The per-token budget is ~4x the measured prefill rate, so
# a hang is still caught, just not a slow machine.
DEFAULT_TIMEOUT = 3600.0
TIMEOUT_S_PER_TOKEN = 3.0
TIMEOUT_OVERHEAD_S = 600.0


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


def select_prompts(ref: str, prompt_set: str) -> list[dict]:
    """Resolve a prompt set against the reference `meta.json`.

    Two shapes live in that file. `meta["singles"]` carries the prompt
    text inline; `meta["longs"]` carries only bookkeeping
    (`file`/`chars`/`tokens`/`depth`) and the text sits beside it in
    `<ref>/<file>.txt`, read verbatim — trailing newline included, since
    that is what was tokenized to produce the recorded token count.

    Token counts are free here: longs record theirs, and singles are
    covered by `meta["tokenized_prompts"]`, so no tokenizer is loaded.
    """
    if prompt_set not in PROMPT_SETS:
        fail(f"unknown prompt set {prompt_set!r}; expected one of "
             f"{', '.join(PROMPT_SETS)}")
    meta_path = os.path.join(ref, "meta.json")
    if not os.path.isfile(meta_path):
        fail(f"no meta.json under {ref}")
    with open(meta_path) as f:
        meta = json.load(f)

    tokens_by_prompt = {
        e["prompt"]: len(e["ids"]) for e in meta.get("tokenized_prompts") or []
    }
    entries: list[dict] = []

    if prompt_set in ("singles", "all"):
        singles = meta.get("singles") or []
        if not singles:
            fail(f"{meta_path} lists no singles")
        for e in singles:
            entries.append({
                "file": e["file"],
                "kind": "single",
                "prompt": e["prompt"],
                "tokens": tokens_by_prompt.get(e["prompt"]),
            })

    if prompt_set in ("longs", "all"):
        longs = meta.get("longs") or []
        if not longs:
            fail(f"{meta_path} lists no longs, so --set {prompt_set} has "
                 f"nothing to fingerprint at chunk length. Point --ref at a "
                 f"reference dir that has them.")
        for e in longs:
            name = e["file"]
            path = os.path.join(ref, f"{name}.txt")
            if not os.path.isfile(path):
                fail(f"{meta_path} lists {name} but its text is missing "
                     f"({path}); the long prompts live in .txt files next to "
                     f"the manifest, not inline")
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if not text:
                fail(f"{path} is empty")
            entries.append({
                "file": name,
                "kind": "long",
                "prompt": text,
                "prompt_file": f"{name}.txt",
                "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "chars": len(text),
                "tokens": e.get("tokens"),
            })

    names = [e["file"] for e in entries]
    if len(set(names)) != len(names):
        fail(f"{meta_path} repeats a prompt file name in set {prompt_set}: "
             f"{sorted({n for n in names if names.count(n) > 1})}")
    return entries


def describe(entry: dict) -> str:
    """One-line label. Never echoes a 15 KB long prompt into the log."""
    tokens = entry.get("tokens")
    if entry["kind"] == "long":
        tok = f", {tokens} tok" if tokens else ""
        return f"{entry['prompt_file']} ({entry['chars']} chars{tok})"
    tok = f" ({tokens} tok)" if tokens else ""
    return f"{entry['prompt']!r}{tok}"


def prompt_timeout(args, entry: dict) -> float:
    """Seconds to allow this prompt. Flat if --timeout was given."""
    if args.timeout is not None:
        return float(args.timeout)
    tokens = entry.get("tokens") or 0
    return max(DEFAULT_TIMEOUT, TIMEOUT_OVERHEAD_S + tokens * TIMEOUT_S_PER_TOKEN)


def run_logits(args, prompt: str, timeout: float, label: str) -> str:
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
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        # Exit 2, not 1: an unfinished prompt produced no bytes, so there
        # is nothing to call identical or different. Raise --timeout (or
        # drop it entirely to get the token-scaled default) and re-run.
        fail(f"ramvamp timed out after {timeout:.0f}s on {label} — no output, "
             f"so this is 'could not run', not a comparison. Re-run with a "
             f"larger --timeout.")
    if result.returncode != 0:
        fail(f"ramvamp exited {result.returncode} on {label}\n"
             f"stderr:\n{result.stderr[-2000:]}")
    return result.stdout


def manifest_record(entry: dict, blob: bytes, digest: str) -> dict:
    """What a capture directory says about one prompt.

    Singles keep the historical shape (`file`/`prompt`/`bytes`/`sha256`)
    so old and new manifests read the same way. Longs record where the
    text came from and its digest instead of inlining 15 KB of Wikipedia
    into the manifest — the payload already echoes the prompt back, and
    the digest is what proves the same fixture was used on both sides.
    """
    rec = {"file": entry["file"], "kind": entry["kind"]}
    if entry["kind"] == "long":
        rec.update({
            "prompt_file": entry["prompt_file"],
            "prompt_sha256": entry["prompt_sha256"],
            "chars": entry["chars"],
        })
    else:
        rec["prompt"] = entry["prompt"]
    rec.update({
        "tokens": entry.get("tokens"),
        "bytes": len(blob),
        "sha256": digest,
    })
    return rec


def capture(args, out_dir: str, quiet: bool = False) -> dict:
    entries = select_prompts(args.ref, args.prompt_set)
    os.makedirs(out_dir, exist_ok=True)
    records = []
    t_start = time.time()
    for entry in entries:
        name, label = entry["file"], describe(entry)
        t0 = time.time()
        stdout = run_logits(args, entry["prompt"], prompt_timeout(args, entry),
                            f"[{name}] {label}")
        blob = stdout.encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        with open(os.path.join(out_dir, f"{name}.json"), "wb") as f:
            f.write(blob)
        records.append(manifest_record(entry, blob, digest))
        if not quiet:
            print(f"[{name}] {digest[:16]}...  {len(blob):>9} B  "
                  f"({time.time() - t0:.0f}s)  {label}")
    manifest = {
        "tool": "scripts/bitident.py",
        "captured": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_commit": git_commit(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "model": os.path.abspath(args.rvmp),
        "reference": os.path.abspath(args.ref),
        "top": args.top,
        "prompt_set": args.prompt_set,
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

    # Manifests predating --set have no `prompt_set` and are singles by
    # construction. Fill it in, then check the claim against the records:
    # a manifest that says `singles` while listing a long is corrupt, and
    # trusting either half of it would silently narrow or widen the gate.
    prompt_set = manifest.setdefault("prompt_set", DEFAULT_PROMPT_SET)
    if prompt_set not in PROMPT_SETS:
        fail(f"{path} records prompt_set {prompt_set!r}, which is not one of "
             f"{', '.join(PROMPT_SETS)}")
    kinds = {rec.get("kind", "single") for rec in manifest["prompts"]}
    expected = {"singles": {"single"}, "longs": {"long"},
                "all": {"single", "long"}}[prompt_set]
    if not kinds <= expected:
        fail(f"{path} records prompt_set {prompt_set!r} but lists prompts of "
             f"kind {sorted(kinds)}; the manifest contradicts itself and "
             f"cannot be used as a baseline. Re-capture it.")
    return manifest


def assert_same_prompt_set(base_dir: str, base: dict,
                           new_dir: str, new: dict) -> None:
    """Both sides must fingerprint the same prompts, or neither counts.

    Comparing a longs capture against a singles baseline would match the
    files they share and report PASS, having never looked at the prompts
    that were the point. That is a harness misuse, so it is exit 2 rather
    than a gate failure.
    """
    if base["prompt_set"] != new["prompt_set"]:
        fail(f"prompt-set mismatch: baseline {base_dir} fingerprints "
             f"{base['prompt_set']!r}, comparand {new_dir} fingerprints "
             f"{new['prompt_set']!r}. These are not comparable; capture the "
             f"comparand with --set {base['prompt_set']}.")
    base_files = sorted(r["file"] for r in base["prompts"])
    new_files = sorted(r["file"] for r in new["prompts"])
    if base_files != new_files:
        missing = sorted(set(base_files) - set(new_files))
        extra = sorted(set(new_files) - set(base_files))
        fail(f"prompt-set mismatch within {base['prompt_set']!r}: "
             f"{len(base_files)} prompts in the baseline, {len(new_files)} in "
             f"the comparand"
             + (f"; missing {missing}" if missing else "")
             + (f"; unexpected {extra}" if extra else "")
             + ". A comparison over a subset is not a pass.")

    # Same prompt names, but are they the same prompt *text*? The long
    # fixtures live in files that can be edited; if long_02.txt changed
    # between the two captures, every logit legitimately differs and the
    # runtime gets blamed for it.
    new_by_file = {r["file"]: r for r in new["prompts"]}
    for rec in base["prompts"]:
        got = new_by_file[rec["file"]]
        for field in ("prompt", "prompt_sha256"):
            if field in rec and rec[field] != got.get(field):
                fail(f"prompt text for {rec['file']} differs between "
                     f"{base_dir} and {new_dir} ({field}); the fixture "
                     f"changed, so any logit difference would not be the "
                     f"runtime's. Re-capture the baseline.")


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

    # Prompt set, same rule as --top: the baseline decides. Silence when
    # nothing was asked for, an error when something contradictory was —
    # `--set longs` against a singles baseline is a mistake worth naming,
    # not a request to quietly fingerprint 8 prompts.
    if args.set_explicit and args.prompt_set != base["prompt_set"]:
        fail(f"--set {args.prompt_set} contradicts baseline {base_dir}, which "
             f"was captured over {base['prompt_set']!r}. A baseline's prompt "
             f"set is part of its identity: compare it as captured, or "
             f"capture a new baseline with --set {args.prompt_set}.")
    if args.prompt_set != base["prompt_set"]:
        print(f"note: baseline fingerprints {base['prompt_set']!r}, using that "
              f"prompt set instead of {args.prompt_set!r}", file=sys.stderr)
        args.prompt_set = base["prompt_set"]

    if args.dry_run:
        print(f"dry run: would compare {base_dir} "
              f"(set {base['prompt_set']}, top {base['top']}, captured "
              f"{base['captured']}, commit {base['git_commit']})")
        print_selection(args)
        return 0

    if args.new:
        new_dir, new_manifest, temp = args.new, load_manifest(args.new), False
        print(f"comparing {base_dir} vs {new_dir} (no run)")
    else:
        new_dir = args.workdir or os.path.join(
            base_dir + ".compare", time.strftime("%Y%m%dT%H%M%S"))
        temp = args.workdir is None
        print(f"re-running current build into {new_dir}")
        new_manifest = capture(args, new_dir)

    assert_same_prompt_set(base_dir, base, new_dir, new_manifest)

    print(f"baseline: {base['git_commit']}  captured {base['captured']}  "
          f"top {base['top']}  set {base['prompt_set']}")
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

    differing = []
    for rec in base["prompts"]:
        name = rec["file"]
        tok = f"  {rec['tokens']:>4} tok" if rec.get("tokens") else ""
        if new_digests[name] == base_digests[name]:
            print(f"[{name}] identical  {base_digests[name][:16]}...{tok}")
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
        print(f"\nbit-identity: PASS  {n}/{n} {base['prompt_set']} prompts "
              f"byte-identical to {base_dir}")
        return 0
    numeric = [d for _, d in differing if d["kind"] == "numeric"]
    kinds = ", ".join(sorted({d["kind"] for _, d in differing}))
    print(f"\nbit-identity: FAIL  {len(differing)}/{n} {base['prompt_set']} "
          f"prompts differ ({kinds}); {len(numeric)} numeric -> the forward "
          f"pass changed")
    return 1


def print_selection(args) -> None:
    """--dry-run: exactly which prompts this invocation would fingerprint."""
    entries = select_prompts(args.ref, args.prompt_set)
    budget = 0.0
    print(f"set {args.prompt_set}: {len(entries)} prompts from "
          f"{os.path.join(args.ref, 'meta.json')} at --top {args.top}")
    for e in entries:
        t = prompt_timeout(args, e)
        budget += t
        print(f"  {e['file']:<12} {e['kind']:<7} "
              f"{(str(e['tokens']) + ' tok') if e.get('tokens') else '? tok':>9}  "
              f"timeout {t:>8.0f}s  {describe(e)[:70]}")
    print(f"  {'':<12} {'total':<7} {'':>9}  timeout {budget:>8.0f}s "
          f"({budget / 3600:.1f} h worst case)")
    inv = binary_identity(args).get("invocation")
    print(f"binary: {inv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("dir", help="baseline directory to write or compare to")
    parser.add_argument("--rvmp", default="models/qwen3.rvmp",
                        help="installed .rvmp model dir")
    parser.add_argument("--ref", default="models/llamacpp-ref/llamacpp_ref",
                        help="reference dir supplying the prompt set "
                             "(meta.json). Not in the repository -- "
                             "`.gitignore` excludes /models/ -- so the default "
                             "resolves to nothing in a fresh clone and the "
                             "dump has to be re-banked from llama.cpp b10217")
    parser.add_argument("--ramvamp",
                        help="ramvamp binary (default: cargo run --release)")
    parser.add_argument("--set", dest="prompt_set", choices=PROMPT_SETS,
                        default=None,
                        help=f"which prompts to fingerprint (default "
                             f"{DEFAULT_PROMPT_SET}): 'singles' = the 8 short "
                             f"prompts, 'longs' = the 512/1891/3492-token "
                             f"fixtures that cross prefill chunk boundaries, "
                             f"'all' = both. compare mode adopts the "
                             f"baseline's set; passing one that contradicts "
                             f"it is an error")
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
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the prompt set, then exit 0 "
                             "without loading the model")
    parser.add_argument("--timeout", type=float, default=None,
                        help=f"flat per-prompt ramvamp timeout (s). Default is "
                             f"token-scaled: max({DEFAULT_TIMEOUT:.0f}, "
                             f"{TIMEOUT_OVERHEAD_S:.0f} + "
                             f"{TIMEOUT_S_PER_TOKEN:.0f}*tokens), so a 3492-token "
                             f"long gets ~3 h instead of the 1 h a flat default "
                             f"would give it. A timeout is exit 2, never a pass")
    args = parser.parse_args()

    # Remember whether --set was actually typed: compare mode adopts the
    # baseline's set when it was not, and refuses when it was and disagrees.
    args.set_explicit = args.prompt_set is not None
    if args.prompt_set is None:
        args.prompt_set = DEFAULT_PROMPT_SET

    if args.mode == "capture":
        if args.dry_run:
            print(f"dry run: would capture into {args.dir}")
            print_selection(args)
            return 0
        if os.path.isdir(args.dir) and os.listdir(args.dir) and not args.force:
            fail(f"{args.dir} exists and is not empty (use --force)")
        manifest = capture(args, args.dir)
        print(f"\ncaptured {len(manifest['prompts'])} {manifest['prompt_set']} "
              f"prompts at --top {manifest['top']} in "
              f"{manifest['elapsed_s']:.0f}s -> {args.dir}")
        print(f"commit {manifest['git_commit']}")
        return 0
    return compare(args, args.dir)


if __name__ == "__main__":
    sys.exit(main())
