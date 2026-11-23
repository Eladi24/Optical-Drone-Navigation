#!/usr/bin/env python3
"""
Why does the frozen config's retrieval fail on AerialVL? A diagnostic, not a tuning loop (2026-11-23).

The Tier-0 smoke of the frozen pipeline on AerialVL av2 missed the true reference crop in the top 10 on 15/15
frames. Three suspects, varied ONE AT A TIME here with the C++-mirroring query (uavvisloc_grid.make_query,
'northup', heading 90, GSD from the calibration):
  1. field of view: an AerialVL frame covers only ~200 m (vs ~245-300 m on UAV-VisLoc), so the query crop is
     200 m while the frozen reference crops are 300 m -> ref crop size 300 (frozen) vs 200 (footprint-matched);
  2. domain: fine-tuned-on-UAV-VisLoc dense DINOv2 + its vocab (frozen config) vs frozen ImageNet DINOv2-S + its
     own vocab (models/vlad_vocab_frozen_k64.bin);
  3. content: reported per segment (av2 village start vs the rest, av1 dusk).
Recall@k = the GT's nearest 200 m grid cell is in the top k (same protocol as eval_retrieval_recall.py).

    python3 -I scripts/aerialvl_retrieval_diag.py [--stride 10]
"""
import argparse
import csv
import math
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import proto_vlad_retrieval as pv  # noqa: E402
import uavvisloc_grid as grid  # noqa: E402

COORDS = REPO / "Datasets" / "aerialvl_coordinates_range.csv"
MAP = REPO / "Images" / "map_clean_aerialvl_large.png"
GSD_TXT = REPO / "Datasets" / "AerialVL" / "visloc_format" / "gsd_calibration.txt"
MODELS = REPO / "models"
CONFIGS = {  # name -> (dense weights, vocab .bin)
    "finetuned (frozen config)": (str(MODELS / "dinov2_s_finetuned_dense.onnx"), MODELS / "vlad_vocab_finetuned_k64.bin"),
    "ImageNet DINOv2 (no fine-tune)": ("none", MODELS / "vlad_vocab_frozen_k64.bin"),
}


def read_vocab(path):
    raw = Path(path).read_bytes()
    magic, k, d = struct.unpack("<iii", raw[:12])
    assert magic == 0x44414C56, path
    return np.frombuffer(raw[12:], dtype="<f4").reshape(k, d).astype(np.float32)


def recall(ranks, ks=(1, 5, 10)):
    r = np.array(ranks)
    return " ".join(f"R@{k} {np.mean(r < k) * 100:5.1f}%" for k in ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10, help="use every Nth frame")
    args = ap.parse_args()

    bounds = grid.load_bounds("aerialvl_large", str(COORDS))
    map_img = cv2.imread(str(MAP))
    mpp = grid.map_mpp(map_img, bounds)
    glats, glngs = grid.generate_grid(*bounds, grid_spacing_m=200.0)
    gsd = {l.split()[0]: float(l.split()[1].split("=")[1]) for l in open(GSD_TXT)}
    print(f"map mpp {mpp:.3f}, {len(glats)} grid cells; GSD {gsd}")

    queries = []  # (seq, frame_idx, query_img, gt_cell, segment)
    for seq in ("av2", "av1"):
        rows = list(csv.DictReader(open(REPO / "Datasets" / "UAV_VisLoc_dataset" / seq / f"{seq}.csv", newline="")))
        for i in range(0, len(rows), args.stride):
            r = rows[i]
            frame = cv2.imread(str(REPO / "Datasets" / "UAV_VisLoc_dataset" / seq / "drone" / r["filename"]))
            q = grid.make_query(frame, "northup", gsd[seq], mpp, heading_deg=float(r["Phi1"]), crop_size_m=300.0)
            cell = grid.nearest_grid_index(float(r["lat"]), float(r["lon"]), glats, glngs)
            seg = (f"{seq} first 100" if i < 100 else f"{seq} rest") if seq == "av2" else \
                  (f"{seq} first half" if i < len(rows) // 2 else f"{seq} second half")
            queries.append((seq, i, q, cell, seg))
    print(f"{len(queries)} queries (every {args.stride}th frame); query crop {queries[0][2].shape[:2]} px "
          f"= {queries[0][2].shape[0] * mpp:.0f} m\n")

    for name, (weights, vocab_path) in CONFIGS.items():
        emb = pv.DenseEmbedder(weights)
        vocab = read_vocab(vocab_path)
        qv = np.stack([pv.vlad_encode(emb.embed_dense(q), vocab) for _, _, q, _, _ in queries])
        for crop_m in (300.0, 200.0):
            refs = []
            for lat, lng in zip(glats, glngs):
                c = grid.crop_at(map_img, lat, lng, bounds, crop_size_m=crop_m)
                refs.append(pv.vlad_encode(emb.embed_dense(c), vocab) if c is not None and c.size else
                            np.zeros(qv.shape[1], np.float32))
            sims = qv @ np.stack(refs).T
            ranks = [int((sims[j] > sims[j, cell]).sum()) for j, (_, _, _, cell, _) in enumerate(queries)]
            tag = "frozen 300 m" if crop_m == 300 else "footprint 200 m"
            print(f"{name:32s} ref crops {tag:16s} ALL  {recall(ranks)}")
            for seg in dict.fromkeys(s for *_, s in queries):
                idx = [j for j, q in enumerate(queries) if q[4] == seg]
                print(f"{'':32s} {'':16s}           {seg:18s} n={len(idx):2d} {recall([ranks[j] for j in idx])}")
        print()


if __name__ == "__main__":
    main()
