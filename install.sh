#!/usr/bin/env bash
# ramvamp installer. Downloads the latest prebuilt release tarball and puts
# both binaries, `ramvamp` and `ramvamp-repack`, on your PATH.
#
# This installs the runtime only. It does not download a model, because that is
# a 17.35 GiB fetch and it is yours to start: `ramvamp-repack install`.
#
# Usage:
#   curl -sSL https://y0sif.github.io/ramvamp/install.sh | bash
#   curl -sSL https://raw.githubusercontent.com/y0sif/ramvamp/main/install.sh | bash
#   ./install.sh
#
# Environment:
#   RAMVAMP_VERSION=v0.1.0   Pin to a specific tag (default: latest release)
#   RAMVAMP_INSTALL_DIR=DIR  Where the two binaries go. Default /usr/local/bin,
#                            falling back to ~/.local/bin when there is neither
#                            root nor sudo.
#
# To build from source instead, see docs/install.md. That needs Rust 1.88; a
# prebuilt binary needs no toolchain at all.

set -euo pipefail

# Colors.
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "  ${GREEN}${BOLD}$1${RESET} $2"; }
warn()  { echo -e "  ${YELLOW}$1${RESET}"; }
error() { echo -e "  ${RED}$1${RESET}"; }
step()  { echo -e "\n${BOLD}[$1/$TOTAL] $2${RESET}"; }

TOTAL=5
REPO="y0sif/ramvamp"
ARTIFACT="ramvamp-linux-x86_64.tar.gz"

echo -e "\n${BOLD}ramvamp installer${RESET}: 26-30B MoE models in about 3 GB of RAM\n"

# ── Refuse anything that is not Linux x86_64 ────────────────────────────
#
# Not a packaging gap. io_uring is a Linux kernel interface and the expert
# streamer is built on it; the fast kernels are AVX2 plus F16C, which is
# x86_64. Elsewhere the code compiles and runs on fallback paths that no
# published figure describes, so there is nothing honest to install.

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
    error "Unsupported platform: ${OS} ${ARCH}"
    echo "  ramvamp is Linux x86_64 only, on purpose:"
    echo ""
    echo "    - the expert streamer is io_uring, which is a Linux interface"
    echo "    - the fast kernels are AVX2 plus F16C, which is x86_64"
    echo ""
    echo "  A binary for another target would run on the slow fallback paths and"
    echo "  none of the published numbers would describe it. Nothing installed."
    echo ""
    echo "  See https://github.com/${REPO}#install"
    exit 1
fi

# ── Step 1: Check the host ──────────────────────────────────────────────

step 1 "Checking the host..."

for tool in curl tar; do
    if ! command -v "$tool" &>/dev/null; then
        error "Missing required tool: $tool"
        echo "  Install it with your package manager and re-run this script."
        exit 1
    fi
done

info "Platform:" "Linux x86_64"

# io_uring landed in kernel 5.1. Losing it costs speed, not correctness: the
# streamer falls back to pread, and the ~3 GB memory contract rides on O_DIRECT
# rather than on io_uring, so the budget survives the fallback.
KERNEL="$(uname -r)"
kmajor="${KERNEL%%.*}"
krest="${KERNEL#*.}"
kminor="${krest%%.*}"
kminor="${kminor%%[!0-9]*}"

if [[ "$kmajor" =~ ^[0-9]+$ ]] && [[ "$kminor" =~ ^[0-9]+$ ]] &&
   { [ "$kmajor" -gt 5 ] || { [ "$kmajor" -eq 5 ] && [ "$kminor" -ge 1 ]; }; }; then
    info "Kernel:" "$KERNEL (io_uring available)"
else
    warn "Kernel $KERNEL is older than 5.1, which is where io_uring landed."
    echo "  ramvamp will fall back to pread: correct, but slower than any"
    echo "  figure in the README. The load banner will tell you which mode you got."
fi

# AVX2 and F16C are detected at runtime and there is a scalar fallback, so this
# is a warning and not a refusal.
if grep -qw avx2 /proc/cpuinfo 2>/dev/null && grep -qw f16c /proc/cpuinfo 2>/dev/null; then
    info "CPU:" "AVX2 and F16C present"
else
    warn "This CPU does not report both AVX2 and F16C."
    echo "  ramvamp runs on the scalar fallback kernels, which are much slower."
fi

# Advisory only: an NVMe device existing does not mean the model will land on
# it. Decode reads up to about 1.1 GB of expert weights per token, so the drive
# is the throughput.
if compgen -G "/sys/class/nvme/nvme*" &>/dev/null; then
    info "Storage:" "NVMe device present (put the model on it)"
else
    warn "No NVMe device found on this machine."
    echo "  ramvamp reads up to about 1.1 GB of expert weights per decode token,"
    echo "  so the drive is the throughput. It runs on SATA or a spinning disk,"
    echo "  just far below the published figures."
fi

# ── Step 2: Resolve the release tag ─────────────────────────────────────

step 2 "Resolving release tag..."

if [ -n "${RAMVAMP_VERSION:-}" ]; then
    TAG="$RAMVAMP_VERSION"
    info "Pinned:" "$TAG (RAMVAMP_VERSION)"
else
    # Ask the releases API, with no jq dependency. Unauthenticated callers get
    # 60 requests an hour per IP, so fall back to following the
    # /releases/latest redirect to its canonical tag URL when that runs out.
    TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        | head -n1) || TAG=""

    if [ -z "$TAG" ]; then
        TAG=$(curl -sSL -o /dev/null -w '%{url_effective}' \
            "https://github.com/${REPO}/releases/latest" \
            | sed 's|.*/tag/||' | tr -d '[:space:]') || TAG=""
    fi

    if [ -z "$TAG" ] || [ "$TAG" = "latest" ]; then
        error "Could not resolve the latest release tag."
        echo "  Set RAMVAMP_VERSION=v0.1.0 (or similar) and re-run."
        exit 1
    fi
    info "Latest:" "$TAG"
fi

# ── Step 3: Download and extract the tarball ────────────────────────────

step 3 "Downloading prebuilt tarball..."

URL="https://github.com/${REPO}/releases/download/${TAG}/${ARTIFACT}"
info "URL:" "$URL"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! curl -fsSL -o "$TMP/$ARTIFACT" "$URL"; then
    error "Download failed."
    echo "  Check that release ${TAG} has artifact ${ARTIFACT}:"
    echo "    https://github.com/${REPO}/releases/tag/${TAG}"
    exit 1
fi

SIZE=$(stat -c %s "$TMP/$ARTIFACT" 2>/dev/null || echo 0)
TENTHS=$(( (SIZE * 10 + 524288) / 1048576 ))
info "Downloaded:" "$(( TENTHS / 10 )).$(( TENTHS % 10 )) MiB"

# There is no checksum file to check against, so verify what is verifiable:
# that this is a real gzip tarball, that it holds both binaries, and that they
# are the version the tag claims.
if ! tar tzf "$TMP/$ARTIFACT" &>/dev/null; then
    error "The download is not a valid gzip tarball."
    echo "  A proxy or captive portal may have returned an error page instead."
    exit 1
fi

tar xzf "$TMP/$ARTIFACT" -C "$TMP"

for bin in ramvamp ramvamp-repack; do
    if [ ! -x "$TMP/$bin" ]; then
        error "Extracted tarball is missing an executable $bin."
        echo "  Report this against release ${TAG}:"
        echo "    https://github.com/${REPO}/issues"
        exit 1
    fi
done

GOT=$("$TMP/ramvamp" --version 2>/dev/null | head -n1 | awk '{print $NF}') || GOT=""

if [ -z "$GOT" ]; then
    warn "The downloaded binary would not run here."
    echo "  It is built against the glibc on ubuntu-latest; an older distro may"
    echo "  be too far back. Building from source works: see docs/install.md."
elif [ "$GOT" = "${TAG#v}" ]; then
    info "Verified:" "reports $GOT, matching $TAG"
else
    warn "Binary reports version $GOT but the tag is $TAG."
fi

# ── Step 4: Install both binaries ───────────────────────────────────────

step 4 "Installing binaries..."

# Where an existing install lives, so an upgrade can say so and a shadowed one
# can be called out rather than silently ignored.
PREV=$(command -v ramvamp 2>/dev/null || true)
PREV_VER=""
if [ -n "$PREV" ]; then
    PREV_VER=$("$PREV" --version 2>/dev/null | head -n1 | awk '{print $NF}') || PREV_VER=""
fi

DEST="${RAMVAMP_INSTALL_DIR:-/usr/local/bin}"

# Writability of the nearest existing ancestor, since DEST itself may not exist.
writable() {
    local d="$1"
    while [ ! -e "$d" ] && [ "$d" != "/" ]; do d="$(dirname "$d")"; done
    [ -w "$d" ]
}

SUDO=""
if writable "$DEST"; then
    :
elif command -v sudo &>/dev/null; then
    SUDO="sudo"
    info "Elevating:" "sudo is needed to write $DEST"
elif [ -z "${RAMVAMP_INSTALL_DIR:-}" ]; then
    DEST="$HOME/.local/bin"
    warn "/usr/local/bin is not writable and sudo is not available."
    echo "  Falling back to $DEST."
else
    error "$DEST is not writable and sudo is not available."
    echo "  Set RAMVAMP_INSTALL_DIR to somewhere you can write, and re-run."
    exit 1
fi

# shellcheck disable=SC2086  # $SUDO is deliberately empty when not needed.
$SUDO install -Dm755 "$TMP/ramvamp"        "$DEST/ramvamp"
# shellcheck disable=SC2086
$SUDO install -Dm755 "$TMP/ramvamp-repack" "$DEST/ramvamp-repack"

info "Installed:" "$DEST/ramvamp"
info "Installed:" "$DEST/ramvamp-repack"

if [ -n "$PREV" ] && [ "$PREV" != "$DEST/ramvamp" ]; then
    warn "Another ramvamp is installed at $PREV."
    echo "  Whichever comes first on PATH wins. Remove the old one, or reorder PATH."
elif [ -n "$PREV_VER" ] && [ "$PREV_VER" != "${TAG#v}" ]; then
    info "Upgraded:" "$PREV_VER to ${TAG#v}"
fi

case ":$PATH:" in
    *":$DEST:"*) ;;
    *)
        warn "$DEST is not on your PATH."
        echo "  Add it, for example:"
        echo ""
        echo "    export PATH=\"$DEST:\$PATH\""
        ;;
esac

# ── Step 5: Point at the model install, and do not start it ─────────────

step 5 "Next: install a model..."

echo ""
echo "  The runtime is in place, and it cannot generate anything yet. ramvamp"
echo "  streams a model from disk and no model is bundled."
echo ""
echo -e "  Qwen3-30B-A3B Q4_K_M is the v0 model, and installing it is a"
echo -e "  ${BOLD}17.35 GiB download${RESET}. This script will not start that for you."
echo "  When you are ready, on a path that lives on an NVMe SSD:"
echo ""
echo -e "    ${BOLD}ramvamp-repack install --output ~/models/qwen3-30b-a3b.rvmp${RESET}"
echo ""
echo "  It streams the source GGUF straight into the packed layout, so it needs"
echo "  17.35 GiB free and not twice that. Then:"
echo ""
echo "    ramvamp chat  --model ~/models/qwen3-30b-a3b.rvmp --tui"
echo "    ramvamp serve --model ~/models/qwen3-30b-a3b.rvmp --port 8080"
echo ""
echo "  Requirements, I/O modes and the repacker's other subcommands:"
echo "    https://github.com/${REPO}/blob/main/docs/install.md"

echo -e "\n${GREEN}${BOLD}Installation complete!${RESET}\n"
