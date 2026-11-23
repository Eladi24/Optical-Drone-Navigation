"""Generates the C++ <-> Python VLAD parity fixture used by tests/test_vlad.cpp.

Uses the project's REAL Python implementations -- scripts/export_vlad_vocab.py's
write_bin (vocabulary file format) and scripts/proto_vlad_retrieval.py's vlad_encode
(the reference encoder) -- on small seeded random data, so the C++ vlad::readVocabulary
and vlad::encode are checked against exactly what the Python side produces.

    source ../CV_IP/cv_env/bin/activate
    python3 -I tests/fixtures/make_vlad_fixture.py

Outputs (committed, a few KB):
    tests/fixtures/vlad_small_vocab.bin   'VLAD' vocab file, k=8, D=16
    tests/fixtures/vlad_small_case.bin    int32 P, D, k; then P*D float32 patches; then k*D float32 expected
"""
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import export_vlad_vocab as ev  # noqa: E402
import proto_vlad_retrieval as pv  # noqa: E402

K, D, P = 8, 16, 50


def unit_rows(a):
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def main():
    rng = np.random.default_rng(20261123)
    centroids = unit_rows(rng.standard_normal((K, D)))
    # Patches clustered around a subset of centroids (leaves some centroids empty, which
    # exercises the zero-block path) plus noise, then unit-normalized like the C++ side.
    owners = rng.choice(K - 2, size=P)
    patches = unit_rows(centroids[owners] + 0.15 * rng.standard_normal((P, D)))

    vocab_path = HERE / "vlad_small_vocab.bin"
    ev.write_bin(centroids, str(vocab_path))
    # Encode against the centroids exactly as written (write_bin re-normalizes with +1e-9).
    stored = np.frombuffer(vocab_path.read_bytes()[12:], dtype="<f4").reshape(K, D)
    expected = pv.vlad_encode(patches, stored).astype(np.float32)

    # The C++ test relies on centroids 6 and 7 receiving no patches: check the REAL assignment.
    assigned = set(np.argmin(((patches[:, None, :] - stored[None]) ** 2).sum(2), axis=1).tolist())
    assert not ({K - 2, K - 1} & assigned), f"centroids {K-2},{K-1} must stay empty, got {assigned}"

    case_path = HERE / "vlad_small_case.bin"
    with open(case_path, "wb") as f:
        f.write(struct.pack("<iii", P, D, K))
        f.write(patches.astype("<f4").tobytes())
        f.write(expected.astype("<f4").tobytes())
    print(f"wrote {case_path} ({case_path.stat().st_size} bytes), empty centroids: "
          f"{sorted(set(range(K)) - assigned)}")


if __name__ == "__main__":
    main()
