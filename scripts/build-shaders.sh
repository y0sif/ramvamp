#!/bin/sh
# Compile the spike's GLSL compute shaders to SPIR-V, or check that the
# committed SPIR-V still matches its source.
#
# Both the .comp and the .spv are committed: the runtime loads the .spv with
# `include_bytes!`, so a build never needs glslc on the machine. That is the
# convenience, and it is also the hazard -- nothing in a normal `cargo build`
# reads the .comp at all, so an edited shader whose .spv was not regenerated
# compiles, links and runs the OLD code, and every timing it produces is of a
# shader nobody is looking at.
#
#   scripts/build-shaders.sh           compile and overwrite the .spv
#   scripts/build-shaders.sh --check   compile to a temp file, compare, and
#                                      exit non-zero on any mismatch
#
# `--check` is local-only enforcement on purpose: the CI workflow states that
# the tree has no C dependencies and installs no system packages, and glslc
# comes from shaderc, which is one. Re-run this after editing a shader and
# commit the regenerated .spv alongside it.
#
# Needs `glslc` from the `shaderc` package. Record the version in the raw
# experiment log alongside any byte-identity claim: `glslc --version` prints
# both the glslc and the shaderc release, and SPIR-V output is only
# reproducible against a stated toolchain.
set -eu

check=0
case "${1-}" in
    --check) check=1 ;;
    "") ;;
    *)
        echo "usage: $0 [--check]" >&2
        exit 2
        ;;
esac

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
shaders="$root/crates/vulkan-spike/shaders"

command -v glslc >/dev/null 2>&1 || {
    echo "build-shaders: glslc not found (pacman -S shaderc)" >&2
    exit 1
}

compile() {
    glslc -O -fshader-stage=compute --target-env=vulkan1.3 "$1" -o "$2"
}

if [ "$check" -eq 1 ]; then
    tmp=$(mktemp -d)
    # shellcheck disable=SC2064 # $tmp is expanded now on purpose.
    trap "rm -rf '$tmp'" EXIT INT TERM
    # SPIR-V output is only reproducible against a stated toolchain, so the
    # version goes in the output of the check that asserts byte identity.
    echo "build-shaders --check: glslc $(glslc --version | sed -n '1p'), shaderc $(glslc --version | sed -n '2p')"
    status=0
    for src in "$shaders"/*.comp; do
        out="${src%.comp}.spv"
        fresh="$tmp/$(basename "$out")"
        compile "$src" "$fresh"
        if [ ! -f "$out" ]; then
            echo "MISSING  $out (no committed SPIR-V for $(basename "$src"))" >&2
            status=1
        elif cmp -s "$out" "$fresh"; then
            echo "ok       $(basename "$out") matches a fresh build of $(basename "$src")"
        else
            echo "MISMATCH $out is not a build of $(basename "$src")" >&2
            echo "         the committed SPIR-V is stale; run $0 and commit the result" >&2
            status=1
        fi
    done
    exit "$status"
fi

for src in "$shaders"/*.comp; do
    out="${src%.comp}.spv"
    echo "glslc $src -> $out"
    compile "$src" "$out"
done
