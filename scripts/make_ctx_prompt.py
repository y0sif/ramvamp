#!/usr/bin/env python3
"""Cut a prompt fixture of an exact token count out of a reference text.

The phase-8 decode sweep needs a prompt at each rung of a context ladder
(64, 512, 1024, 2048, 4096 tokens). Three of those rungs did not exist on
disk. This script makes them, and it makes them the way a measurement
fixture has to be made:

* **Derived, not invented.** The bytes come from truncating a reference
  text (`models/llamacpp-ref/llamacpp_ref/long_02.txt` by default).
  Nothing is generated, sampled, or hand-written. That text is **not in
  the repository**: `.gitignore` excludes `/models/`, so a fresh clone has
  nothing to truncate and can regenerate none of these fixtures. The
  manifest therefore records the *source's* own SHA-256 and byte length
  beside each fixture's, which is what reproducibility can honestly mean
  here -- anyone holding a source file with that hash regenerates
  identical bytes, and anyone holding a different one can tell that they
  are not about to. (The fixtures are not committed either; `/scratch/`
  is excluded on the same grounds.)
* **Counted, not guessed.** The token count is verified with the real
  tokenizer via `ramvamp tokenize`, not estimated from a bytes-per-token
  ratio. A binary search over the character prefix finds the boundary, and
  a bounded linear scan around it lands on the *exact* count. If the exact
  count is unreachable, this fails loudly rather than writing a fixture
  that is close.
* **Recorded.** Byte length and SHA-256 -- of the source text, of the file
  written, and of the string `cold_bench.py` will actually deliver -- are
  printed and written to a manifest, the same provenance EXP-021 recorded
  for `scratch/ctx4k/p4k.txt` ("sha256 68582aae37b9... over 17,000 bytes").

Two constraints come straight from `scripts/cold_bench.py`, and this script
enforces both so a fixture can never fail three cold runs into an overnight:

1. `read_prompt_file` strips **exactly one** trailing terminator, counting
   a trailing CRLF as one and not two, because a trailing `\\n` is a token
   of its own and would shift the tok/s being measured. So the *delivered*
   prompt is not always the file, and the count that matters is the
   delivered one. A cut landing on a newline would leave N tokens on disk
   and hand ramvamp N-1, while the manifest and the sweep runbook publish
   the rung as N -- an unlabelled off-by-one in a context curve, which is
   the exact failure this script exists to prevent, so it is not left to
   luck. A cut that would not survive the strip unchanged is rejected by
   the search; the *delivered* string is what gets counted; and file and
   delivered bytes are compared before anything is written. Fixtures
   therefore carry **no** trailing terminator (matching `long_00.txt` and
   `p4k.txt`, neither of which has one) -- but both hashes and both byte
   counts are reported anyway, because a reader who cannot see both cannot
   tell which prompt was measured.
2. The prompt becomes one argv element of the `ramvamp generate` command,
   and Linux caps a single argv element at `MAX_ARG_STRLEN` = 32 pages.
   A 4096-token prompt is ~17 KB, so every rung here has headroom, but the
   ceiling is checked rather than assumed.

Usage:

    python3 scripts/make_ctx_prompt.py --targets 64,1024,2048
    python3 scripts/make_ctx_prompt.py --targets 1024 --out scratch/x.txt

Exit codes: 0 all fixtures written, 2 could not produce one (missing
binary, missing source, exact count unreachable, over the argv ceiling).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

# Linux caps a *single* argv element at 32 pages, separately from ARG_MAX.
# Same constant, same reason, as scripts/cold_bench.py: the prompt is one
# argv element of the ramvamp workload, and exceeding it dies with a bare
# E2BIG that names no argument.
PAGE = 4096
MAX_ARG_STRLEN = 32 * PAGE

# How far around the binary-search boundary to scan for an exact hit. Token
# counts are non-decreasing in prefix length but not strictly so: one more
# character can merge two tokens into one or split one into two, so the
# boundary can jump straight over the target. At ~4.3 bytes/token on this
# reference text, 64 characters is ~15 tokens of slack in each direction.
SCAN_RADIUS = 64

DEFAULT_SOURCE = "models/llamacpp-ref/llamacpp_ref/long_02.txt"
DEFAULT_MODEL = "models/qwen3.rvmp"
DEFAULT_OUTDIR = "scratch/phase8/prompts"


def fail(msg: str) -> None:
    """Exit 2: the fixture could not be produced. Never write a partial one."""
    print(f"make_ctx_prompt: {msg}", file=sys.stderr)
    raise SystemExit(2)


def resolve_binary(explicit: str | None, root: str) -> tuple[str, str]:
    """The ramvamp binary to tokenize with, and a note about which it is.

    Deliberately does **not** build. A missing release binary during an
    overnight prep is worth thirty seconds of the user's attention, not
    fifteen minutes of `cargo build --release` starting behind their back.
    A debug binary is accepted -- the tokenizer is the same tokenizer and
    the token count is the same count -- but the caller is told, because
    "which binary produced this fixture" belongs in the provenance.
    """
    if explicit:
        if not os.path.isfile(explicit):
            fail(f"--ramvamp {explicit} does not exist")
        return explicit, "explicit"
    release = os.path.join(root, "target/release/ramvamp")
    debug = os.path.join(root, "target/debug/ramvamp")
    if os.path.isfile(release):
        return release, "release"
    if os.path.isfile(debug):
        print(f"NOTE: {release} does not exist; using the DEBUG binary\n"
              f"      {debug}\n"
              f"      The tokenizer is identical, so the token counts are\n"
              f"      correct. Not building: a build is not this script's\n"
              f"      job to start unasked.", file=sys.stderr)
        return debug, "debug"
    fail(f"neither {release} nor {debug} exists. Build one "
         f"(`cargo build --release`) or pass --ramvamp; this script will "
         f"not start a build on your behalf.")
    raise AssertionError("unreachable")


def strip_one_terminator(text: str) -> str:
    """Remove exactly one trailing line terminator, as `cold_bench.py` does.

    `cold_bench.read_prompt_file` strips a trailing CRLF as *one*
    terminator and a trailing LF as one, and nothing else. Both places
    this script needs that rule -- reading the source text, and computing
    the prompt cold_bench.py will deliver -- go through this one function,
    so they cannot drift from each other or from the consumer. Stripping
    only `"\\n"` here would leave a dangling `"\\r"` on a cut that landed
    on a CRLF: a delivered hash for a string ramvamp never sees.
    """
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


def read_source(path: str) -> tuple[str, dict]:
    """The reference text and its provenance, read as `cold_bench.py` reads it.

    Bytes, then an explicit UTF-8 decode -- never text-mode `open`, whose
    universal-newline translation would silently rewrite every CRLF to LF
    and make the fixture differ from the file it claims to be derived from.
    Exactly one trailing terminator is stripped so that truncating this
    text and truncating the delivered prompt mean the same thing.

    The returned provenance hashes the source **as it is on disk**, before
    that strip, because the point of recording it is to identify the file
    someone else would have to hold -- and it is read once here rather than
    reopened per fixture.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        fail(f"cannot read --source {path}: {e}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        fail(f"--source {path} is not valid UTF-8: {e}")
    text = strip_one_terminator(text)
    if not text:
        fail(f"--source {path} is empty after stripping one trailing newline")
    return text, {"source": path,
                  "source_bytes": len(raw),
                  "source_sha256": hashlib.sha256(raw).hexdigest()}


class Tokenizer:
    """`ramvamp tokenize` as a callable, with the counts memoized.

    Every count costs a process spawn and a tokenizer load (~0.4 s with the
    release binary), and the search asks for the same prefix lengths more
    than once, so the cache is not an optimization detail -- it is most of
    the runtime.
    """

    def __init__(self, binary: str, model: str) -> None:
        self.binary = binary
        self.model = model
        self._cache: dict[int, int] = {}
        self.calls = 0

    def count_prefix(self, text: str, length: int) -> int:
        """Tokens in `text[:length]`, as the real tokenizer counts them."""
        if length in self._cache:
            return self._cache[length]
        n = self.count_text(text[:length])
        self._cache[length] = n
        return n

    def count_text(self, chunk: str) -> int:
        if not chunk:
            return 0
        size = len(chunk.encode("utf-8"))
        if size > MAX_ARG_STRLEN:
            fail(f"a {size}-byte candidate is over the {MAX_ARG_STRLEN}-byte "
                 f"MAX_ARG_STRLEN ceiling on one argv element, so it cannot "
                 f"be tokenized this way -- and cold_bench.py would reject "
                 f"it for the same reason. Pick a smaller --target.")
        self.calls += 1
        proc = subprocess.run(
            [self.binary, "tokenize", "--model", self.model, "--prompt", chunk],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            fail(f"`ramvamp tokenize` exited {proc.returncode}:\n"
                 f"{proc.stderr.strip()[:800]}")
        for line in proc.stdout.splitlines():
            # The CLI prints `tokens: N` (crates/cli/src/main.rs, fn tokenize).
            if line.startswith("tokens: "):
                return int(line.split(":", 1)[1].strip())
        fail(f"`ramvamp tokenize` printed no `tokens:` line. Its output "
             f"surface changed; this script parses it. Got:\n"
             f"{proc.stdout.strip()[:800]}")
        raise AssertionError("unreachable")


def find_exact_prefix(tok: Tokenizer, text: str, target: int) -> int:
    """The prefix length that *delivers* exactly `target` tokens.

    Binary search for the smallest prefix that reaches the target, then a
    bounded scan either side of that boundary, because the count is
    non-decreasing in prefix length but not strictly increasing: adding one
    character can retokenize the tail and jump the count by two.

    A candidate must clear two bars, not one. Its token count must be
    exactly `target`, and it must survive `cold_bench.py`'s
    strip-one-trailing-terminator unchanged -- i.e. it must not end on a
    newline. A cut ending on a newline is not merely untidy: the file would
    hold `target` tokens while ramvamp prefilled `target - 1`, and the
    rung would be published as the number that was never measured. Such a
    cut is skipped rather than salvaged, and nothing is lost by skipping
    it: the prefix one character shorter is the string that cut would have
    delivered, and the scan reaches it anyway.
    """
    total_tokens = tok.count_prefix(text, len(text))
    if target > total_tokens:
        fail(f"--target {target} exceeds the whole source text, which is "
             f"{total_tokens} tokens. Truncation cannot add tokens; use a "
             f"longer --source.")

    # Counted separately from the accept/reject decision so a failure can
    # say *which* bar the near misses fell at.
    skipped_on_newline = 0

    def usable(cand: int) -> bool:
        nonlocal skipped_on_newline
        if tok.count_prefix(text, cand) != target:
            return False
        chunk = text[:cand]
        if strip_one_terminator(chunk) != chunk:
            skipped_on_newline += 1
            return False
        return True

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if tok.count_prefix(text, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    boundary = lo
    if usable(boundary):
        return boundary

    # The boundary jumped the target, or landed on a newline. Scan outward
    # from it, nearest first, so the fixture stays as close to the natural
    # cut as possible.
    for delta in range(1, SCAN_RADIUS + 1):
        for cand in (boundary - delta, boundary + delta):
            if 0 <= cand <= len(text) and usable(cand):
                return cand
    detail = (f" ({skipped_on_newline} prefix(es) did tokenize to {target} "
              f"but ended on a newline, which cold_bench.py strips before "
              f"the prompt reaches ramvamp, so they would have delivered "
              f"{target - 1})" if skipped_on_newline else "")
    fail(f"no prefix of --source delivers exactly {target} tokens "
         f"within {SCAN_RADIUS} characters of the search boundary at "
         f"{boundary} (which is {tok.count_prefix(text, boundary)} "
         f"tokens){detail}. "
         f"Try a different --source; a fixture that is 'about' {target} "
         f"tokens would put an unlabelled error into the context curve.")
    raise AssertionError("unreachable")


def write_fixture(path: str, prompt: str) -> dict:
    """Write the fixture and return its provenance record.

    Written with **no** trailing terminator, which is what `long_00.txt`
    and `p4k.txt` do, so the file bytes and the bytes cold_bench.py
    delivers to ramvamp are the same bytes. That equality is checked here
    rather than assumed -- `find_exact_prefix` already refuses a cut that
    would break it, and this is the second lock on the door -- and both
    hashes are reported regardless, because the pair is the only way a
    reader can tell that they agree.
    """
    # What cold_bench.py will actually hand to ramvamp: the file with one
    # trailing terminator stripped, CRLF counted as one.
    delivered = strip_one_terminator(prompt)
    if delivered != prompt:
        fail(f"internal: the prompt for {path} ends in a line terminator, so "
             f"cold_bench.py would deliver {len(delivered)} of its "
             f"{len(prompt)} characters to ramvamp and the recorded token "
             f"count would be one too high. find_exact_prefix should never "
             f"return such a cut.")
    data = prompt.encode("utf-8")
    if len(data) > MAX_ARG_STRLEN:
        fail(f"{path} would be {len(data)} bytes, over the "
             f"{MAX_ARG_STRLEN}-byte MAX_ARG_STRLEN ceiling; cold_bench.py "
             f"would refuse it.")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    delivered_bytes = delivered.encode("utf-8")
    return {
        "path": os.path.abspath(path),
        "file_bytes": len(data),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "delivered_bytes": len(delivered_bytes),
        "delivered_sha256": hashlib.sha256(delivered_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--targets", default="64,1024,2048",
                        help="comma-separated exact token counts to produce "
                             "(default: %(default)s)")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="reference text to truncate (default: "
                             "%(default)s). Not in the repository -- "
                             "`.gitignore` excludes /models/ -- so its own "
                             "SHA-256 and byte length go into the manifest, "
                             "and that hash is what makes a fixture "
                             "regenerable by someone else")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="installed .rvmp dir whose tokenizer decides the "
                             "count (default: %(default)s)")
    parser.add_argument("--ramvamp", default=None,
                        help="ramvamp binary providing `tokenize` (default: "
                             "target/release/ramvamp, falling back to "
                             "target/debug/ramvamp with a warning)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="directory for ctx<N>.txt fixtures "
                             "(default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help="write a single fixture to this exact path "
                             "(requires exactly one --targets value)")
    parser.add_argument("--manifest", default=None,
                        help="JSON provenance record (default: "
                             "<outdir>/MANIFEST.json; ignored with --out)")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        targets = [int(t) for t in args.targets.split(",") if t.strip()]
    except ValueError:
        fail(f"--targets {args.targets!r} is not a comma-separated integer list")
    if not targets:
        fail("--targets is empty")
    if any(t < 1 for t in targets):
        fail("--targets must all be >= 1; a zero-token prompt has no prefill")
    if args.out and len(targets) != 1:
        fail(f"--out names one file but --targets has {len(targets)} values; "
             f"use --outdir for more than one")

    source = args.source if os.path.isabs(args.source) \
        else os.path.join(root, args.source)
    model = args.model if os.path.isabs(args.model) \
        else os.path.join(root, args.model)
    if not os.path.isdir(model):
        fail(f"--model {model} is not a directory (the installed .rvmp dir)")

    binary, kind = resolve_binary(args.ramvamp, root)
    text, source_meta = read_source(source)
    tok = Tokenizer(binary, model)

    print(f"source   {source}")
    print(f"         {source_meta['source_bytes']} bytes on disk, "
          f"sha256 {source_meta['source_sha256']}")
    print(f"         {len(text)} chars, "
          f"{len(text.encode('utf-8'))} bytes after one trailing newline is "
          f"stripped")
    print(f"tokenizer {binary} ({kind}) against {model}")
    print(f"argv ceiling MAX_ARG_STRLEN = {MAX_ARG_STRLEN} bytes "
          f"({MAX_ARG_STRLEN // PAGE} pages)")
    print()

    records = []
    for target in targets:
        length = find_exact_prefix(tok, text, target)
        prompt = text[:length]
        # Re-count from scratch, and count the string ramvamp is actually
        # handed rather than the one on disk. The two differ by whatever
        # `cold_bench.read_prompt_file` strips, and it is the delivered
        # count that gets published as the rung -- so verifying the file's
        # count would verify a number nobody measures. Bypasses the search
        # cache deliberately: a check that reuses the chooser's own
        # arithmetic checks nothing.
        delivered = strip_one_terminator(prompt)
        actual = tok.count_text(delivered)
        if actual != target:
            fail(f"internal: the prefix of {length} chars delivers {actual} "
                 f"tokens, wanted {target}. The file would say {target} and "
                 f"ramvamp would prefill {actual}.")
        out = args.out or os.path.join(args.outdir, f"ctx{target}.txt")
        if not os.path.isabs(out):
            out = os.path.join(root, out)
        rec = write_fixture(out, prompt)
        rec.update({"tokens": target, "prefix_chars": length,
                    **source_meta,
                    "tokenizer_binary": binary, "tokenizer_kind": kind,
                    "model": model})
        records.append(rec)
        print(f"{target:>5} tokens  ->  {rec['path']}")
        print(f"           bytes  {rec['file_bytes']}  "
              f"(delivered {rec['delivered_bytes']})")
        print(f"           sha256 {rec['file_sha256']}")
        print(f"           delivered sha256 {rec['delivered_sha256']}")
        print(f"           cut at {length} chars of source; "
              f"{rec['file_bytes'] / target:.2f} bytes/token")
        print()

    if not args.out:
        manifest = args.manifest or os.path.join(args.outdir, "MANIFEST.json")
        if not os.path.isabs(manifest):
            manifest = os.path.join(root, manifest)
        os.makedirs(os.path.dirname(manifest), exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"fixtures": records}, f, indent=2)
            f.write("\n")
        print(f"manifest {manifest}")

    print(f"{tok.calls} tokenizer invocations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
