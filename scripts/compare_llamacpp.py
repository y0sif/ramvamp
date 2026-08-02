#!/usr/bin/env python3
"""Compare ramvamp against llama.cpp on identical GGUF weights.

Validation gates 3 and 4 of docs/architecture.md ("Validation protocol"):

  greedy  -- run `ramvamp generate --greedy` and `llama-cli --temp 0` on the
             same prompt and report how far the emitted texts agree
             (greedy streams are expected to diverge eventually from fp
             reordering; the match length is reported, not gated).
  logits  -- run `ramvamp logits` and query llama-server's /completion with
             n_probs for the same prompt; report top-1 agreement and a KL
             divergence over the union of the two top-N sets.

             Truncation caveat: both sides only expose their top-N
             logprobs, so each distribution is renormalized over the union
             of the two top-N token sets before the KL is computed. Mass
             outside the union is ignored; this underestimates the true KL
             when the two sides spread mass differently in the tail. The
             full-vocabulary KL gate (mean KL <= 1e-3) needs a dump of all
             151936 logits on both sides and comes later.

Python stdlib only. If the llama.cpp binaries are absent the script exits
with a clear message instead of failing cryptically.

Examples:
  scripts/compare_llamacpp.py greedy \
      --rvmp models/qwen3.rvmp --gguf ~/models/Qwen3-30B.gguf \
      --prompt "The capital of France is" --max-new 16
  scripts/compare_llamacpp.py logits \
      --rvmp models/qwen3.rvmp --gguf ~/models/Qwen3-30B.gguf \
      --prompt "The capital of France is" --top 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def fail(message: str) -> "None":
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def resolve_binary(explicit: str | None, names: list[str], what: str) -> str | None:
    """Find a binary: explicit path first, then PATH under common names."""
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        return None
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def run_ramvamp(args: argparse.Namespace, subcommand: list[str]) -> str:
    """Run the ramvamp CLI (via a binary or `cargo run --release`)."""
    if args.ramvamp:
        cmd = [args.ramvamp]
    else:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "-p",
            "ramvamp",
            "--",
        ]
    cmd += subcommand
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    if result.returncode != 0:
        fail(
            f"ramvamp exited {result.returncode}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
    return result.stdout


def probe_flags(binary: str) -> str:
    """The binary's --help text, for probing flag spellings across versions."""
    try:
        result = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout + result.stderr
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# greedy mode
# ---------------------------------------------------------------------------


def cmd_greedy(args: argparse.Namespace) -> int:
    llama_cli = resolve_binary(args.llama_cli, ["llama-cli", "main"], "llama-cli")
    if llama_cli is None:
        fail(
            "llama-cli not found (looked at --llama-cli and PATH). "
            "Install llama.cpp and re-run; the ramvamp side alone can be "
            "exercised with `ramvamp generate --greedy`."
        )
    if not os.path.isfile(args.gguf):
        fail(f"GGUF file not found: {args.gguf}")

    ours = run_ramvamp(
        args,
        [
            "generate",
            "--model",
            args.rvmp,
            "--prompt",
            args.prompt,
            "--greedy",
            "--max-new",
            str(args.max_new),
            "--skip-hashes",
        ],
    ).rstrip("\n")

    helptext = probe_flags(llama_cli)
    # -c bounds the KV cache; see the --ctx help text for why it is vital.
    cmd = [llama_cli, "-m", args.gguf, "-c", str(args.ctx), "-p", args.prompt, "--temp", "0", "-n", str(args.max_new)]
    # Flag spellings move between llama.cpp releases; probe --help.
    if "-no-cnv" in helptext:
        cmd.append("-no-cnv")
    elif "--no-conversation" in helptext:
        cmd.append("--no-conversation")
    if "--no-display-prompt" in helptext:
        cmd.append("--no-display-prompt")
    # Skip the load-time warmup that touches every weight byte: on hosts
    # with less RAM than the model, it triggers OOM killers (earlyoom).
    if "--no-warmup" in helptext:
        cmd.append("--no-warmup")
    if "--seed" in helptext:
        cmd += ["--seed", "42"]
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=args.timeout, check=False
    )
    if result.returncode != 0:
        fail(f"llama-cli exited {result.returncode}\nstderr:\n{result.stderr[-2000:]}")
    theirs = result.stdout
    # Without --no-display-prompt llama-cli echoes the prompt; strip it.
    if "--no-display-prompt" not in cmd and args.prompt in theirs:
        theirs = theirs.split(args.prompt, 1)[1]
    theirs = theirs.strip("\n")
    # llama-cli appends end-of-generation markers on some builds.
    for marker in ("[end of text]", "<|im_end|>", "<|endoftext|>"):
        theirs = theirs.replace(marker, "")
    theirs = theirs.rstrip()

    match = 0
    for a, b in zip(ours, theirs):
        if a != b:
            break
        match += 1
    full = ours == theirs
    print(json.dumps(
        {
            "mode": "greedy",
            "prompt": args.prompt,
            "max_new": args.max_new,
            "ramvamp_text": ours,
            "llamacpp_text": theirs,
            "match_chars": match,
            "ramvamp_chars": len(ours),
            "llamacpp_chars": len(theirs),
            "full_match": full,
        },
        indent=2,
        ensure_ascii=False,
    ))
    # Match length is reported, not gated (see docs/architecture.md gate 4).
    return 0


# ---------------------------------------------------------------------------
# logits mode
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_health(base: str, proc: subprocess.Popen, deadline: float) -> None:
    while time.time() < deadline:
        if proc.poll() is not None:
            fail(f"llama-server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    fail("llama-server did not become healthy in time")


def extract_top_probs(payload: dict) -> list[dict]:
    """Normalize llama-server /completion probability shapes across versions.

    Returns a list of {"id": int|None, "text": str, "logprob": float} for
    the FIRST predicted position, most probable first.
    """
    cp = payload.get("completion_probabilities")
    if not cp:
        fail(f"no completion_probabilities in llama-server response: {list(payload)}")
    first = cp[0]
    # Newer servers: {"id", "token", "logprob", "top_logprobs": [{"id", "token", "logprob"}]}
    if "top_logprobs" in first:
        rows = first["top_logprobs"]
        out = [
            {"id": r.get("id"), "text": r.get("token", ""), "logprob": float(r["logprob"])}
            for r in rows
        ]
        return out
    # Older servers: {"content", "probs": [{"tok_str", "prob"}]}
    if "probs" in first:
        out = []
        for r in first["probs"]:
            prob = float(r.get("prob", 0.0))
            logprob = math.log(prob) if prob > 0 else -math.inf
            out.append(
                {"id": r.get("id"), "text": r.get("tok_str", r.get("token", "")), "logprob": logprob}
            )
        return out
    fail(f"unrecognized completion_probabilities shape: {list(first)}")
    return []  # unreachable


def cmd_logits(args: argparse.Namespace) -> int:
    llama_server = resolve_binary(
        args.llama_server, ["llama-server", "server"], "llama-server"
    )
    if llama_server is None:
        fail(
            "llama-server not found (looked at --llama-server and PATH). "
            "Install llama.cpp and re-run; the ramvamp side alone can be "
            "exercised with `ramvamp logits`."
        )
    if not os.path.isfile(args.gguf):
        fail(f"GGUF file not found: {args.gguf}")

    ours = json.loads(
        run_ramvamp(
            args,
            [
                "logits",
                "--model",
                args.rvmp,
                "--prompt",
                args.prompt,
                "--top",
                str(args.top),
                "--skip-hashes",
            ],
        )
    )
    our_top = ours["top"]  # [{token_id, logit, logprob, text}]

    port = args.port or free_port()
    base = f"http://127.0.0.1:{port}"
    server_cmd = [
        llama_server,
        "-m",
        args.gguf,
        # -c bounds the KV cache; see the --ctx help text for why it is vital.
        "-c",
        str(args.ctx),
        # See the greedy path: full-weight warmup OOM-kills small-RAM hosts.
        "--no-warmup",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    print(f"+ {' '.join(server_cmd)}", file=sys.stderr)
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_health(base, proc, time.time() + args.timeout)
        body = json.dumps(
            {
                "prompt": args.prompt,
                "n_predict": 1,
                "n_probs": args.top,
                "temperature": 0.0,
                "samplers": [],
            }
        ).encode()
        req = urllib.request.Request(
            base + "/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            payload = json.loads(resp.read())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    their_top = extract_top_probs(payload)

    # Keying: token ids when the server provides them. Text keying is NOT a
    # sound fallback: ramvamp renders partial-UTF-8 byte tokens as U+FFFD
    # while llama.cpp returns raw byte fragments, so text keys silently fail
    # to match and inflate the reported KL. Fail loudly instead.
    have_ids = all(r.get("id") is not None for r in their_top)
    if not have_ids:
        sys.exit(
            "error: llama-server response has no token ids in its logprobs; "
            "text-keyed comparison is unsound (U+FFFD vs raw byte fragments "
            "would inflate KL). Use a llama.cpp build that returns ids."
        )

    def key_ours(row: dict):
        return row["token_id"] if have_ids else row["text"]

    def key_theirs(row: dict):
        return row["id"] if have_ids else row["text"]

    ours_lp = {key_ours(r): r["logprob"] for r in our_top}
    theirs_lp = {key_theirs(r): r["logprob"] for r in their_top}

    top1_ours = key_ours(our_top[0])
    top1_theirs = key_theirs(their_top[0])
    top1_agree = top1_ours == top1_theirs

    # KL(theirs || ours) over the union of the two top-N sets, each side
    # renormalized over that union (see the module docstring's caveat).
    union = set(ours_lp) | set(theirs_lp)
    floor = -30.0  # logprob assigned to tokens missing from a side's top-N
    p_raw = {k: math.exp(theirs_lp.get(k, floor)) for k in union}
    q_raw = {k: math.exp(ours_lp.get(k, floor)) for k in union}
    p_sum = sum(p_raw.values())
    q_sum = sum(q_raw.values())
    kl = 0.0
    for k in union:
        p = p_raw[k] / p_sum
        q = q_raw[k] / q_sum
        if p > 0 and q > 0:
            kl += p * math.log(p / q)

    overlap = len(set(ours_lp) & set(theirs_lp))
    print(json.dumps(
        {
            "mode": "logits",
            "prompt": args.prompt,
            "top_n": args.top,
            "keyed_by": "token_id" if have_ids else "token_text",
            "top1_ramvamp": top1_ours,
            "top1_llamacpp": top1_theirs,
            "top1_agree": top1_agree,
            "topn_overlap": overlap,
            "kl_theirs_vs_ours_union_renorm": kl,
            "note": (
                "KL is over the union of the two top-N sets with both sides "
                "renormalized; tail mass outside the union is ignored"
            ),
            "ramvamp_top": our_top,
            "llamacpp_top": their_top,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0 if top1_agree else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rvmp", required=True, help="installed .rvmp model dir")
    common.add_argument("--gguf", required=True, help="the pinned source GGUF file")
    common.add_argument(
        "--ramvamp",
        help="path to the ramvamp binary (default: cargo run --release -p ramvamp)",
    )
    common.add_argument("--prompt", default="The capital of France is")
    common.add_argument(
        "--timeout", type=float, default=600.0, help="per-command timeout (s)"
    )
    common.add_argument(
        "--ctx",
        type=int,
        default=4096,
        help="llama.cpp context size; REQUIRED to be small — without -c, "
        "llama.cpp defaults to the model's native 262144 context and "
        "allocates a ~24 GB KV cache (OOM-kill on 16 GB hosts)",
    )

    g = sub.add_parser("greedy", parents=[common], help="greedy text comparison")
    g.add_argument("--llama-cli", help="path to llama-cli (default: search PATH)")
    g.add_argument("--max-new", type=int, default=16)
    g.set_defaults(func=cmd_greedy)

    l = sub.add_parser("logits", parents=[common], help="top-logprob comparison")
    l.add_argument("--llama-server", help="path to llama-server (default: search PATH)")
    l.add_argument("--top", type=int, default=20)
    l.add_argument("--port", type=int, help="llama-server port (default: free port)")
    l.set_defaults(func=cmd_logits)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
