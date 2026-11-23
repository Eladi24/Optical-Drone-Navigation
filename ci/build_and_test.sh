#!/usr/bin/env bash
# Configure, build and test exactly as CI does. The same script runs in GitHub Actions
# (.github/workflows/ci.yml) and locally (ci/run_in_container.sh, or directly on the host).
#
# Environment (all optional):
#   BUILD_DIR          build directory                      (default: build-ci)
#   BUILD_TYPE         CMake build type                      (default: Release)
#   ENABLE_ONNX        ON/OFF for DRONENAV_ENABLE_ONNX       (default: OFF -- see below)
#   TEST_LABELS        ctest -L regex                        (default: unit|integration)
#   EXTRA_CMAKE_ARGS   appended to the configure line
#
# ONNX is OFF by default: since ~2026-11-18 github.com/microsoft/onnxruntime (and its release
# downloads) are private while Microsoft investigates malicious content, so a clean machine cannot
# fetch the pinned 1.28.0 tarball. We deliberately do not mirror that binary while the
# investigation is open. The ONNX-only tests need the (gitignored) models anyway, so CI loses only
# compile coverage of the five ONNX stage sources. Re-enable once upstream publishes a trusted release.
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-ci}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_ONNX="${ENABLE_ONNX:-OFF}"
TEST_LABELS="${TEST_LABELS:-unit|integration}"

launcher=()
if command -v ccache >/dev/null 2>&1; then
    launcher=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
generator=()
if command -v ninja >/dev/null 2>&1; then
    generator=(-G Ninja)
fi

echo "::group::configure (${CXX:-c++}, ${BUILD_TYPE}, ONNX=${ENABLE_ONNX})"
# shellcheck disable=SC2086  # EXTRA_CMAKE_ARGS is intentionally word-split
cmake -S . -B "${BUILD_DIR}" "${generator[@]}" "${launcher[@]}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DDRONENAV_ENABLE_ONNX="${ENABLE_ONNX}" \
    -DDRONENAV_BUILD_TESTS=ON \
    ${EXTRA_CMAKE_ARGS:-}
echo "::endgroup::"

echo "::group::build"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-$(nproc)}"
echo "::endgroup::"

echo "::group::test (labels: ${TEST_LABELS})"
ctest --test-dir "${BUILD_DIR}" -L "${TEST_LABELS}" --output-on-failure \
      --output-junit "${PWD}/${BUILD_DIR}/junit.xml"
echo "::endgroup::"

if command -v ccache >/dev/null 2>&1; then
    ccache --show-stats || true
fi
