#!/usr/bin/env bash
# Runs ci/build_and_test.sh inside a clean ubuntu:24.04 container (the same OS image and apt
# packages as the GitHub runner), on exactly the files git would commit. This catches what a
# dev machine hides: missing-from-git files, cached downloads, newer local libraries
# (this machine has a source-built OpenCV 4.10; the runner gets apt's 4.6).
#
#   ci/run_in_container.sh [gcc|clang]     (default gcc)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

compiler="${1:-gcc}"
case "${compiler}" in
    gcc)   cc=gcc;   cxx=g++ ;;
    clang) cc=clang; cxx=clang++ ;;
    *) echo "usage: $0 [gcc|clang]" >&2; exit 2 ;;
esac

# Tracked + untracked-but-not-ignored files: what a fresh checkout of the next commit would contain.
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT
git ls-files -z --cached --others --exclude-standard | tar --null -T - -cf - | tar -xf - -C "${stage}"

docker run --rm -v "${stage}:/src:ro" -e CC="${cc}" -e CXX="${cxx}" -e BUILD_JOBS="${BUILD_JOBS:-2}" \
    ubuntu:24.04 bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends \
            cmake ninja-build ccache clang g++ ca-certificates libopencv-dev libcurl4-openssl-dev >/dev/null
        cp -a /src /work && cd /work
        echo "OpenCV $(pkg-config --modversion opencv4), $(${CXX} --version | head -1)"
        ci/build_and_test.sh'
