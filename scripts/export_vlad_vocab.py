#!/usr/bin/env python3
"""
Builds the VLAD vocabulary (k-means centroids over dev-flight reference-crop
DINOv2 patch descriptors) and writes it as a small binary the C++
OnnxVladRetrieval stage reads directly:

    int32  magic  = 0x44414C56  ('VLAD', little-endian)
    int32  k      = number of centroids
    int32  D      = per-patch descriptor dim (384)
    float32[k*D]  centroids, row-major, L2-normalized

The k-means itself is done by scripts/proto_vlad_retrieval.build_vocabulary()
(shared, so the C++ stage and the Python recall harness use an identical
vocabulary). Vocabulary depends only on the dense backbone + k + the dev crops,
so it is rebuilt rarely; models/ is gitignored (like the .onnx files) -- this
script is the reproducible source.

Usage:
    source ../CV_IP/cv_env/bin/activate

    # Fine-tuned dense backbone, k=64 -- the gate default
    python3 scripts/export_vlad_vocab.py \
        --weights models/dinov2_s_finetuned_dense.onnx --clusters 64 \
        --out models/vlad_vocab_finetuned_k64.bin

    # Frozen backbone (training-free variant)
    python3 scripts/export_vlad_vocab.py --weights none --clusters 64 \
        --out models/vlad_vocab_frozen_k64.bin

    # Convert an existing .npy (from a prior proto_vlad_retrieval.py run) without
    # re-running k-means
    python3 scripts/export_vlad_vocab.py --from-npy models/vlad_vocab_fc7ed5ee_k64.npy \
        --out models/vlad_vocab_finetuned_k64.bin
"""
import argparse
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import proto_vlad_retrieval as pv

MAGIC = 0x44414C56  # 'VLAD'


def write_bin(centroids: np.ndarray, out_path: str):
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    k, d = centroids.shape
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<iii", MAGIC, k, d))
        f.write(centroids.astype("<f4").tobytes())
    print(f"wrote {out_path}  (k={k}, D={d}, {Path(out_path).stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="none",
                     help="'none' (frozen DINOv2-S) or a path to a dense .onnx export")
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--vocab-crops-per-flight", type=int, default=150)
    ap.add_argument("--from-npy", default=None,
                     help="skip k-means; convert this existing .npy centroid file instead")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.from_npy:
        centroids = np.load(args.from_npy)
        print(f"loaded {args.from_npy}  {centroids.shape}")
    else:
        tag = "frozen" if args.weights in (None, "none") else \
            hashlib.md5(args.weights.encode()).hexdigest()[:8]
        embedder = pv.DenseEmbedder(args.weights)
        centroids = pv.build_vocabulary(embedder, tag, args.clusters, args.vocab_crops_per_flight)

    write_bin(centroids, args.out)


if __name__ == "__main__":
    main()
