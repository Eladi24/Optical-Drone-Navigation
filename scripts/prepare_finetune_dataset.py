#!/usr/bin/env python3
"""
Stage 1a of the DINOv2-S retrieval fine-tuning pipeline (see
.claude/plans/enumerated-yawning-nygaard.md and CLAUDE.md's Investigation Log
for the full context). Builds a manifest of (drone frame, positive crop
coordinates, hard-negative crop coordinates, easy-negative crop coordinates)
training pairs from the dev+validation UAV-VisLoc flights (01, 03, 04, 08,
10) -- held-out flights (02, 05, 06, 11) are deliberately NEVER touched here,
so scripts/eval_retrieval_recall.py's zero-shot check on them stays honest.

Does NOT pre-crop and save individual crop images -- would duplicate
gigabytes of redundant pixels across overlapping 200m-spaced grid cells.
Instead writes crop COORDINATES; scripts/finetune_retrieval_backbone.py's
Dataset crops on the fly from the already-on-disk satellite map PNGs
(Images/map_clean_uavvisloc_<flight>.png) via scripts/uavvisloc_grid.py.

Hard negatives are mined via a fast HSV-histogram similarity ranking against
the flight's own full reference-crop grid -- deliberately mirroring this
project's existing classical retrieval baseline (HistogramRetrieval.cpp /
GlobalHistogramRetrieval), not a from-scratch design, and far cheaper than a
full ORB re-run per frame. Historical per-frame candidate identities from
past benchmark runs could NOT be reused for this (confirmed by an Explore
pass this session: ORBFeatureEstimator only logs aggregate rank/distance
numbers to telemetry CSVs, never which specific wrong crops scored highly) --
mining fresh here is the only option, not a shortcut skipped out of laziness.

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/prepare_finetune_dataset.py
"""
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
OUT_MANIFEST = CSV_DIR / "finetune_manifest.csv"

TRAIN_FLIGHTS = ["01", "03", "04", "08", "10"]  # dev + validation ONLY -- never held-out
IMAGE_PREFIX = {f: f for f in TRAIN_FLIGHTS}  # UAV-VisLoc convention: prefix == flight number

N_HARD_NEGATIVES = 4
N_EASY_NEGATIVES = 2
HARD_NEG_EXCLUDE_RADIUS_M = 300.0   # grid cells this close to the true positive don't count as
                                      # "wrong" -- they may be visually near-identical to it
EASY_NEG_MIN_DIST_M = 1000.0
HIST_BINS = (30, 32)  # H, S bins -- mirrors HistogramRetrieval.cpp's own HSV histogram params


def hsv_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def local_xy(lat, lng, lat0, lng0):
    mlng = grid.m_per_deg_lng(lat0)
    return (lng - lng0) * mlng, (lat - lat0) * grid.M_PER_DEG_LAT


def process_flight(flight, writer, rng):
    map_path = IMAGES_DIR / f"map_clean_uavvisloc_{flight}.png"
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    map_img = cv2.imread(str(map_path))
    if map_img is None:
        print(f"  SKIP flight {flight}: could not read {map_path}")
        return 0
    bounds = grid.load_bounds(f"satellite{flight}.tif")
    lt_lat, lt_lon, rb_lat, rb_lon = bounds

    grid_lats, grid_lngs = grid.generate_grid(lt_lat, lt_lon, rb_lat, rb_lon)
    n_cells = len(grid_lats)
    print(f"  flight {flight}: {n_cells} grid cells")

    # Precompute one HSV histogram per grid cell, once -- reused for every frame in this flight.
    cell_hists = []
    cell_valid = np.zeros(n_cells, dtype=bool)
    for i in range(n_cells):
        crop = grid.crop_at(map_img, grid_lats[i], grid_lngs[i], bounds)
        if crop is None or crop.size == 0:
            cell_hists.append(None)
            continue
        cell_hists.append(hsv_hist(crop))
        cell_valid[i] = True
    print(f"    {cell_valid.sum()}/{n_cells} cells produced a valid crop")

    gt_rows = list(csv.DictReader(open(gt_path, newline="")))
    lat0, lng0 = float(gt_rows[0]["Lat"]), float(gt_rows[0]["Lng"])  # arbitrary local-xy origin
    grid_x, grid_y = local_xy(grid_lats, grid_lngs, lat0, lng0)

    n_written = 0
    for row in gt_rows:
        lat, lng, label = float(row["Lat"]), float(row["Lng"]), row["Label"]
        drone_path = DATASET_DIR / flight / "drone" / label
        if not drone_path.exists():
            continue
        frame = cv2.imread(str(drone_path))
        if frame is None:
            continue

        pos_idx = grid.nearest_grid_index(lat, lng, grid_lats, grid_lngs)
        if not cell_valid[pos_idx]:
            continue

        fx, fy = local_xy(lat, lng, lat0, lng0)
        dist_m = np.hypot(grid_x - fx, grid_y - fy)

        # Hard negatives: top-scoring WRONG cells by HSV histogram similarity
        # (mirrors HistogramRetrieval.cpp), excluding anything too close to the
        # true positive to count as genuinely "wrong".
        frame_hist = hsv_hist(cv2.resize(frame, (300, 300)))
        scores = np.full(len(grid_lats), -1.0)
        for i in range(len(grid_lats)):
            if not cell_valid[i] or dist_m[i] < HARD_NEG_EXCLUDE_RADIUS_M:
                continue
            scores[i] = cv2.compareHist(frame_hist, cell_hists[i], cv2.HISTCMP_CORREL)
        hard_idx = np.argsort(-scores)[:N_HARD_NEGATIVES]
        hard_idx = [i for i in hard_idx if scores[i] > -1.0]

        # Easy negatives: random cells far from the positive.
        far_candidates = np.where(cell_valid & (dist_m > EASY_NEG_MIN_DIST_M))[0]
        easy_idx = rng.choice(far_candidates, size=min(N_EASY_NEGATIVES, len(far_candidates)),
                               replace=False) if len(far_candidates) > 0 else []

        hard_str = ";".join(f"{grid_lats[i]:.7f},{grid_lngs[i]:.7f}" for i in hard_idx)
        easy_str = ";".join(f"{grid_lats[i]:.7f},{grid_lngs[i]:.7f}" for i in easy_idx)
        writer.writerow({
            "flight": flight, "drone_image": str(drone_path.relative_to(REPO)),
            "pos_lat": f"{grid_lats[pos_idx]:.7f}", "pos_lng": f"{grid_lngs[pos_idx]:.7f}",
            "hard_negs": hard_str, "easy_negs": easy_str,
        })
        n_written += 1

    print(f"    wrote {n_written}/{len(gt_rows)} frame pairs")
    return n_written


def main():
    rng = np.random.default_rng(42)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["flight", "drone_image", "pos_lat", "pos_lng",
                                                "hard_negs", "easy_negs"])
        writer.writeheader()
        for flight in TRAIN_FLIGHTS:
            print(f"Processing flight {flight}...")
            total += process_flight(flight, writer, rng)
    print(f"\nWrote {total} training pairs to {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
