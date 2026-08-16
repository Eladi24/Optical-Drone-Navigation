#!/usr/bin/env python3
"""
Reference-map quality audit: for a UAV-VisLoc flight, samples N well-separated
ground-truth frames, crops the reference satellite map at each frame's TRUE
coordinate, crops the corresponding drone photo, and reports both a
quantitative content-mismatch signal and a saved visual montage for manual
confirmation.

Why this exists: the "Why Retrieval Fails on Flights 08 and 10" investigation
(see CLAUDE.md) found flight 08's reference map has severe, pervasive content
staleness vs. the drone imagery -- found by an ad hoc visual check that was
never formalized or applied to every flight. Only flights 01/03 got a direct
frame-1 crop-vs-photo sanity check when originally added to the pipeline;
04/08/10 were assigned as validation flights without this check ever being
run. This script makes that check repeatable and applies it to every flight
that's actually in use.

The quantitative signal (HSV 2D histogram correlation, same method and
parameters as GlobalHistogramRetrieval.cpp, for consistency with what the
pipeline's own classical retrieval stage sees) is a TRIAGE aid, not a
verdict -- a low score can mean genuine content mismatch, but can also just
mean different lighting/season/crop framing. This project's own Esri
investigation already learned that lesson the hard way (a naive per-pixel
cloud detector wrongly cleared frames a human eye caught immediately) -- so
this script always renders the visual montage alongside the score. Read the
picture, not just the number.

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/audit_reference_map_quality.py --flight 04
    python3 scripts/audit_reference_map_quality.py --flight 04,08,10 --n-frames 8
"""
import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
CSV_DIR = REPO / "CSV Files"
DATASET_DIR = REPO / "Datasets" / "UAV_VisLoc_dataset"
IMAGES_DIR = REPO / "Images"
OUT_DIR = REPO / "Images" / "map_quality_audit"
CROP_M = 300
HFOV_DEG = 70.0
H_BINS, S_BINS = 30, 32  # matches HistogramRetrieval.cpp exactly


def hsv_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def load_bounds(flight):
    with open(DATASET_DIR / "satellite_ coordinates_range.csv") as f:
        for r in csv.DictReader(f):
            name = r["mapname"].replace("satellite", "").replace(".tif", "")
            if name == flight:
                return {k: float(r[k]) for k in
                        ("LT_lat_map", "LT_lon_map", "RB_lat_map", "RB_lon_map")}
    raise ValueError(f"No bounds found for flight {flight}")


def audit_flight(flight, n_frames=6):
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    tel_path = CSV_DIR / f"telemetry_uavvisloc_{flight}.csv"
    map_path = IMAGES_DIR / f"map_clean_uavvisloc_{flight}.png"
    drone_dir = DATASET_DIR / flight / "drone"

    gt = list(csv.DictReader(open(gt_path)))
    tel = {int(r["Frame"]): r for r in csv.DictReader(open(tel_path))}
    bounds = load_bounds(flight)
    map_img = cv2.imread(str(map_path))
    H, W = map_img.shape[:2]
    mpp_y = (bounds["LT_lat_map"] - bounds["RB_lat_map"]) * 111320 / H
    half_px = int(CROP_M / mpp_y / 2)

    candidate_idxs = np.linspace(0, len(gt) - 1, n_frames + 2).astype(int)
    panels, scores = [], []
    for idx in candidate_idxs:
        row = gt[idx]
        frame = int(row["Frame"])
        drone_path = drone_dir / f"{flight}_{frame:04d}.JPG"
        if not drone_path.exists():
            continue
        lat, lng = float(row["Lat"]), float(row["Lng"])
        px = int((lng - bounds["LT_lon_map"]) / (bounds["RB_lon_map"] - bounds["LT_lon_map"]) * W)
        py = int((lat - bounds["LT_lat_map"]) / (bounds["RB_lat_map"] - bounds["LT_lat_map"]) * H)
        map_crop = map_img[max(0, py - half_px):py + half_px, max(0, px - half_px):px + half_px]
        if map_crop.size == 0:
            continue
        map_crop = cv2.resize(map_crop, (220, 220))

        alt = float(tel[frame]["Alt"])
        drone = cv2.imread(str(drone_path))
        h, w = drone.shape[:2]
        gsd = 2 * alt * math.tan(math.radians(HFOV_DEG / 2)) / w
        take = min(int(round(CROP_M / gsd)), min(h, w))
        cx, cy = w // 2, h // 2
        sx, sy = cx - take // 2, cy - take // 2
        drone_crop = cv2.resize(drone[sy:sy + take, sx:sx + take], (220, 220))

        score = cv2.compareHist(hsv_hist(map_crop), hsv_hist(drone_crop), cv2.HISTCMP_CORREL)
        scores.append((frame, score))

        pair = np.concatenate([map_crop, drone_crop], axis=0)
        cv2.putText(pair, f"f{frame} map", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(pair, f"drone  corr={score:.2f}", (5, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        panels.append(pair)
        if len(panels) == n_frames:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    montage = np.concatenate(panels, axis=1)
    out_path = OUT_DIR / f"audit_flight{flight}.png"
    cv2.imwrite(str(out_path), montage)

    corr_vals = [s for _, s in scores]
    mean_corr = sum(corr_vals) / len(corr_vals)
    low_frac = sum(1 for c in corr_vals if c < 0.3) / len(corr_vals)
    print(f"Flight {flight}: n={len(scores)} sampled frames, mean HSV-hist correlation={mean_corr:.2f}, "
          f"fraction <0.3 (likely mismatch)={low_frac:.0%}")
    for frame, s in scores:
        flag = "  <-- LOW, inspect visually" if s < 0.3 else ""
        print(f"    frame {frame:>5}: corr={s:.2f}{flag}")
    print(f"    montage: {out_path}")
    return mean_corr, low_frac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True, help="Comma-separated flight numbers, e.g. 01,04,08,10")
    ap.add_argument("--n-frames", type=int, default=6)
    args = ap.parse_args()

    for flight in args.flight.split(","):
        audit_flight(flight.strip(), args.n_frames)
        print()


if __name__ == "__main__":
    main()
