#!/usr/bin/env bash
# Full ramvamp-vs-llama.cpp validation on a Kaggle CPU notebook (30 GB RAM).
#
# Setup on Kaggle:
#   1. Upload ramvamp-src.tar.gz (git archive of the repo) as a private
#      dataset and attach it to a CPU notebook (Settings: internet ON).
#   2. In one cell:  !bash /kaggle/input/<dataset-name>/kaggle_validate.sh
#      (or copy this script's contents into a %%bash cell).
#
# Expects: /kaggle/input/*/ramvamp-src.tar.gz  (first match wins).
# Produces: comparison output on stdout; artifacts under /kaggle/working.
set -euo pipefail

WORK=/kaggle/working
SRC_TAR=$(ls /kaggle/input/*/ramvamp-src.tar.gz | head -1)
GGUF_URL="https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/6c6e8692f43e4ca663f7ece8229a1361090d3a4c/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
LLAMA_RELEASE_TAG="${LLAMA_RELEASE_TAG:-latest}"

echo "== host =="
nproc; free -h | head -2; grep -o -m1 'avx2\|avx512' /proc/cpuinfo | sort -u || echo "no AVX2!"

echo "== rust toolchain =="
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  source "$HOME/.cargo/env"
fi

echo "== source =="
mkdir -p "$WORK/ramvamp" && tar -xzf "$SRC_TAR" -C "$WORK/ramvamp"
cd "$WORK/ramvamp"

echo "== build (release) =="
cargo build --release 2>&1 | tail -3

echo "== install model via ramvamp-repack (streams ~17.4 GB) =="
if [ ! -f "$WORK/qwen3.rvmp/manifest.json" ]; then
  ./target/release/ramvamp-repack install --output "$WORK/qwen3.rvmp" --resume \
    || ./target/release/ramvamp-repack install --output "$WORK/qwen3.rvmp"
fi
./target/release/ramvamp-repack verify-install --input "$WORK/qwen3.rvmp"

echo "== fetch source GGUF (17.4 GB) =="
GGUF="$WORK/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
[ -f "$GGUF" ] || curl -L -C - -o "$GGUF" "$GGUF_URL"

echo "== llama.cpp prebuilt binaries =="
if [ ! -x "$WORK/llama/llama-cli" ]; then
  mkdir -p "$WORK/llama" && cd "$WORK/llama"
  if [ "$LLAMA_RELEASE_TAG" = "latest" ]; then
    API_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
  else
    API_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$LLAMA_RELEASE_TAG"
  fi
  ZIP_URL=$(curl -s "$API_URL" | grep -o 'https://[^"]*bin-ubuntu[^"]*x64[^"]*\.zip' | head -1)
  echo "downloading $ZIP_URL"
  curl -L -o llama.zip "$ZIP_URL" && unzip -oq llama.zip
  # Releases nest binaries under build/bin/ in some tags; normalize.
  BIN=$(find . -name llama-cli -type f | head -1)
  ln -sf "$(dirname "$BIN")"/llama-* .
  cd "$WORK/ramvamp"
fi
export PATH="$WORK/llama:$PATH"
llama-cli --version || true

echo "== smoke: ramvamp generate =="
./target/release/ramvamp generate --model "$WORK/qwen3.rvmp" \
  --prompt "The capital of France is" --greedy --max-new 8 --skip-hashes

echo "== comparison: greedy =="
python3 scripts/compare_llamacpp.py greedy \
  --rvmp "$WORK/qwen3.rvmp" --gguf "$GGUF" \
  --ramvamp ./target/release/ramvamp \
  --prompt "The capital of France is" --max-new 16

python3 scripts/compare_llamacpp.py greedy \
  --rvmp "$WORK/qwen3.rvmp" --gguf "$GGUF" \
  --ramvamp ./target/release/ramvamp \
  --prompt "Water is composed of" --max-new 16

python3 scripts/compare_llamacpp.py greedy \
  --rvmp "$WORK/qwen3.rvmp" --gguf "$GGUF" \
  --ramvamp ./target/release/ramvamp \
  --prompt "In Rust, ownership means" --max-new 16

echo "== comparison: logits =="
python3 scripts/compare_llamacpp.py logits \
  --rvmp "$WORK/qwen3.rvmp" --gguf "$GGUF" \
  --ramvamp ./target/release/ramvamp \
  --prompt "The capital of France is"

python3 scripts/compare_llamacpp.py logits \
  --rvmp "$WORK/qwen3.rvmp" --gguf "$GGUF" \
  --ramvamp ./target/release/ramvamp \
  --prompt "In Rust, ownership means"

echo "== DONE — paste everything above back into the session =="
