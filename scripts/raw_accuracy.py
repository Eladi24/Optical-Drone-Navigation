#!/usr/bin/env python3
"""
Ungated per-frame raw accuracy of a C++ run (2026-11-24): over EVERY still in the flight's ground-truth CSV, a
still scores at threshold T if its telemetry row has Measurement_Valid = 1 and the raw (unfused) position is within
T m of GT; missing or invalid rows count as misses. Unlike stills_odometry_fusion.py's "raw anchors" line, this does
not depend on the fusion inlier gate, so it stays comparable when a variant changes the inlier counts.
Reproduces the "raw A@50" figures quoted for the footprint-crop test (av1 fp 32.9%, av2 fp 60.2%).

    python3 scripts/raw_accuracy.py --estimator split_vlad_xfeat_fp --flights av2,av1
"""
import argparse
import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
THRESHOLDS = (20, 50, 100)


def raw_errors(estimator, flight):
    gt_path = REPO / "CSV Files" / f"ground_truth_uavvisloc_{flight}.csv"
    tel_path = REPO / "CSV Files" / f"video_telemetry_{estimator}_uavvisloc_uavvisloc_{flight}.csv"
    gt = {int(r["Frame"]): (float(r["Lat"]), float(r["Lng"])) for r in csv.DictReader(open(gt_path, newline=""))}
    rows = {int(r["Frame"]): r for r in csv.DictReader(open(tel_path, newline=""))}
    errs = []
    for frame, (glat, glng) in sorted(gt.items()):
        r = rows.get(frame)
        if r is None or r["Measurement_Valid"] != "1":
            errs.append(math.inf)
            continue
        lat, lng = float(r["Raw_Lat"]), float(r["Raw_Lng"])
        errs.append(math.hypot((lat - glat) * 111320.0, (lng - glng) * 111320.0 * math.cos(math.radians(glat))))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--flights", required=True, help="comma-separated, e.g. av2,av1 or 01")
    args = ap.parse_args()
    for f in args.flights.split(","):
        errs = raw_errors(args.estimator, f)
        n = len(errs)
        valid = sum(math.isfinite(e) for e in errs)
        acc = " ".join(f"A@{t} {100.0 * sum(e < t for e in errs) / n:6.2f}%" for t in THRESHOLDS)
        print(f"{args.estimator} {f}: stills {n}, valid {valid} | raw {acc}")


if __name__ == "__main__":
    main()
