#!/usr/bin/env python3
"""
Stage 1c / Stage 3 of the DINOv2-S retrieval fine-tuning pipeline (see
.claude/plans/enumerated-yawning-nygaard.md). Offline recall@{1,5,10}
evaluator, no C++ build needed: embeds every reference-crop grid cell and
every ground-truth frame for a flight (via ONNX Runtime, CPU), ranks
candidates by cosine similarity, and checks whether the true grid cell lands
in the top-{1,5,10}.

Reusable for both the frozen baseline (--weights none, live PyTorch DINOv2-S,
matching models/dinov2_s_retrieval.onnx's un-fine-tuned weights exactly) and
any fine-tuned checkpoint exported to ONNX
(scripts/export_retrieval_backbone.py --weights <checkpoint.pt>).

Runs against ALL benchmarked flights, not just the fine-tuning training set:
dev (01, 03), validation (04, 08, 10), and held-out (02, 05, 06, 11) --
held-out flights were never used as fine-tuning data
(scripts/prepare_finetune_dataset.py only touches 01/03/04/08/10), so their
recall@k here is a genuine zero-shot generalization check, directly
comparable to every recall@k number already recorded in CLAUDE.md for
classical retrieval / frozen DeiT-Tiny / plain ORB.

Usage:
    source ../CV_IP/cv_env/bin/activate

    # Frozen ImageNet DINOv2-S (no fine-tune) -- the "before" baseline
    python3 scripts/eval_retrieval_recall.py --weights none

    # A fine-tuned, already-exported ONNX model
    python3 scripts/eval_retrieval_recall.py --weights models/dinov2_s_finetuned_retrieval.onnx
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import uavvisloc_grid as grid

REPO = Path(__file__).parent.parent
CSV_DIR = REPO / "CSV Files"
IMAGES_DIR = REPO / "Images"
DATASET_DIR = REPO / "Datasets" / "UAV_VisLoc_dataset"

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# All 9 benchmarked flights this session's work touches -- deliberately not
# limited to the fine-tuning training set (01/03/04/08/10), since the
# held-out flights (02/05/06/11) are exactly what makes this a real
# generalization check. Flight 07 (telemetry schema anomaly) and 09
# (multi-tile map, unsupported) excluded, same as the Phase 4 gate.
DEV_FLIGHTS = ["01", "03"]
VALIDATION_FLIGHTS = ["04", "08", "10"]
HELD_OUT_FLIGHTS = ["02", "05", "06", "11"]
ALL_FLIGHTS = DEV_FLIGHTS + VALIDATION_FLIGHTS + HELD_OUT_FLIGHTS


class Embedder:
    """Wraps either a live PyTorch DINOv2-S (frozen, --weights none) or an
    ONNX Runtime session (any exported model, frozen or fine-tuned) behind
    one embed(bgr_image) -> np.ndarray(D,) L2-normalized interface, so the
    rest of this script doesn't care which one it's using."""

    def __init__(self, weights):
        self.use_onnx = weights not in (None, "none")
        if self.use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
            print(f"Embedder: ONNX Runtime, model={weights}")
        else:
            import timm
            import torch
            self.torch = torch
            self.model = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                            num_classes=0, img_size=INPUT_SIZE)
            self.model.eval()
            print("Embedder: live PyTorch, frozen ImageNet DINOv2-S (no fine-tune)")

    def _preprocess(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        return chw

    def embed(self, bgr_image):
        x = self._preprocess(bgr_image)
        if self.use_onnx:
            out = self.session.run(["embedding"], {"image": x})[0][0]
        else:
            with self.torch.no_grad():
                out = self.model(self.torch.from_numpy(x)).numpy()[0]
        return out / (np.linalg.norm(out) + 1e-9)


def eval_flight(flight, embedder, queries=("full",), frame_stride=1):
    map_path = IMAGES_DIR / f"map_clean_uavvisloc_{flight}.png"
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    map_img = cv2.imread(str(map_path))
    if map_img is None:
        print(f"  SKIP flight {flight}: could not read {map_path}")
        return None
    bounds = grid.load_bounds(f"satellite{flight}.tif")
    grid_lats, grid_lngs = grid.generate_grid(*bounds)

    crop_embs, valid_idx = [], []
    for i in range(len(grid_lats)):
        crop = grid.crop_at(map_img, grid_lats[i], grid_lngs[i], bounds)
        if crop is None or crop.size == 0:
            continue
        crop_embs.append(embedder.embed(crop))
        valid_idx.append(i)
    crop_embs = np.stack(crop_embs)                       # (N, D)
    valid_lats, valid_lngs = grid_lats[valid_idx], grid_lngs[valid_idx]
    print(f"  flight {flight}: {len(valid_idx)} valid grid cells embedded")

    gt_rows = list(csv.DictReader(open(gt_path, newline="")))
    mpp = grid.map_mpp(map_img, bounds)
    alt = grid.load_altitude_cache(flight) if any(q != "full" for q in queries) else None
    headings = grid.load_headings(flight) if "northup" in queries else {}
    ranks = {q: [] for q in queries}
    for row in gt_rows:
        if int(row["Frame"]) % frame_stride:
            continue
        lat, lng, label = float(row["Lat"]), float(row["Lng"]), row["Label"]
        drone_path = DATASET_DIR / flight / "drone" / label
        frame = cv2.imread(str(drone_path))
        if frame is None:
            continue
        true_idx = grid.nearest_grid_index(lat, lng, valid_lats, valid_lngs)
        gsd = grid.frame_gsd(alt, frame.shape[1]) if alt else None
        for q in queries:
            query = grid.make_query(frame, q, gsd, mpp, headings.get(int(row["Frame"])))
            sims = crop_embs @ embedder.embed(query)
            order = np.argsort(-sims)
            ranks[q].append(int(np.where(order == true_idx)[0][0]) + 1)  # 1-indexed

    for q in queries:
        ranks[q] = np.array(ranks[q])
        recall = {k: float((ranks[q] <= k).mean()) for k in (1, 5, 10)}
        print(f"    [{q:>7}] n={len(ranks[q])}  recall@1={recall[1]:.1%}  recall@5={recall[5]:.1%}  "
              f"recall@10={recall[10]:.1%}")
    return ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="none",
                     help="'none' for the frozen ImageNet baseline (live PyTorch), or a path to an "
                          "exported .onnx model (frozen or fine-tuned)")
    ap.add_argument("--flights", default=None,
                     help="Comma-separated flight numbers to evaluate (default: all 9 benchmarked "
                          "flights -- dev+validation+held-out)")
    ap.add_argument("--queries", default="full",
                     help="Comma-separated query modes from " + ",".join(grid.QUERY_MODES) + ": 'full' = "
                          "legacy whole-photo protocol; 'crop' = the C++ pipeline's GSD-normalised 300 m "
                          "centre crop; 'northup' = that crop with the -heading pre-warp (2026-11-18)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="score every Nth GT frame (consecutive stills overlap heavily)")
    args = ap.parse_args()

    flights = args.flights.split(",") if args.flights else ALL_FLIGHTS
    queries = tuple(args.queries.split(","))
    embedder = Embedder(args.weights)

    per_flight = {}
    for flight in flights:
        print(f"Evaluating flight {flight}...")
        ranks = eval_flight(flight, embedder, queries, args.frame_stride)
        if ranks:
            per_flight[flight] = ranks
    report_pooled(per_flight, queries, f"weights={args.weights}", args.json_out)


def report_pooled(per_flight, queries, title, json_out=None):
    """Pools ranks by split (dev/validation/held-out/ALL) per query mode."""
    import json
    splits = {"dev": DEV_FLIGHTS, "validation": VALIDATION_FLIGHTS, "held-out": HELD_OUT_FLIGHTS}
    print(f"\n{'=' * 70}\n{title}\nPOOLED BY GROUP\n{'=' * 70}")
    out = {"title": title, "per_flight": {}, "pooled": {}}
    for q in queries:
        all_r = []
        for name, members in splits.items():
            lists = [per_flight[f][q] for f in members if f in per_flight]
            if not lists:
                continue
            pooled = np.concatenate(lists); all_r.append(pooled)
            r = {k: float((pooled <= k).mean()) for k in (1, 5, 10)}
            out["pooled"][f"{q}/{name}"] = {"n": int(len(pooled)), **{f"R@{k}": v for k, v in r.items()}}
            print(f"[{q:>7}] {name:>10}  n={len(pooled):5d}  recall@1={r[1]:.1%}  "
                  f"recall@5={r[5]:.1%}  recall@10={r[10]:.1%}")
        if all_r:
            pooled = np.concatenate(all_r)
            r = {k: float((pooled <= k).mean()) for k in (1, 5, 10)}
            out["pooled"][f"{q}/ALL"] = {"n": int(len(pooled)), **{f"R@{k}": v for k, v in r.items()}}
            print(f"[{q:>7}] {'ALL':>10}  n={len(pooled):5d}  recall@1={r[1]:.1%}  "
                  f"recall@5={r[5]:.1%}  recall@10={r[10]:.1%}")
    for f, d in per_flight.items():
        out["per_flight"][f] = {q: {f"R@{k}": float((d[q] <= k).mean()) for k in (1, 5, 10)} for q in queries}
    if json_out:
        json.dump(out, open(json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
