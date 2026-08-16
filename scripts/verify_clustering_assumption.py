#!/usr/bin/env python3
"""
STRATEGY.md Phase 4 pre-check (open question #6 / §8 Phase 4 task 2):

    "Verify the clustering assumption on your own data before building the
    solver: plot retrieval candidates against the ground-truth trajectory and
    confirm correct matches cluster."

The batch/trajectory-level fusion Phase 4 proposes (§3) only pays off if
per-frame raw position guesses are *not* pure noise with respect to the true
trajectory -- i.e. if the population of raw guesses across a flight contains
a detectable "near-truth" mode (even if most frames are wrong), distinct from
guesses that behave like a position drawn uniformly at random from the whole
reference map. If raw guesses are statistically indistinguishable from random
map-wide guesses, no batch estimator can recover signal that was never there.

This script tests exactly that, using data already on disk (plain `orb`
telemetry -- the algorithm with the best full-database recall measured so
far -- against the dense per-frame UAV-VisLoc ground truth, no new pipeline
run needed):

For each flight:
  1. Load ground truth (dense, ~1 point/frame) and ORB raw telemetry.
  2. Compute the actual per-frame raw position error (metres).
  3. Build a null distribution: for each ground-truth frame, draw K points
     uniformly at random from the flight's own satellite-map bounding box
     and compute their distance to that frame's true position. This is
     "what per-frame error would look like if the estimator's guess carried
     zero spatial information about the truth."
  4. Compare actual vs. null: overall mean/median, fraction within a
     near-truth threshold (500m), and a two-sample Kolmogorov-Smirnov test
     (are the two distributions actually different, not just eyeballed).
  5. Save a histogram (actual vs. null) and a spatial scatter (ground-truth
     trajectory + raw guesses, colored by error) per flight for visual
     inspection.

Run from the CV_IP venv (needs numpy/scipy/matplotlib):
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/verify_clustering_assumption.py
"""
import csv
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_ground_truth import load_ground_truth, load_telemetry, haversine_flat_m, m_per_deg_lng, M_PER_DEG_LAT

REPO = Path(__file__).parent.parent
CSV_DIR = REPO / "CSV Files"
BOUNDS_CSV = REPO / "Datasets" / "UAV_VisLoc_dataset" / "satellite_ coordinates_range.csv"
OUT_DIR = REPO / "Images" / "clustering_check"
FLIGHTS = ["01", "03", "04", "08", "10"]
NEAR_TRUTH_THRESHOLD_M = 500.0
NULL_SAMPLES_PER_FRAME = 20
RNG_SEED = 42


def load_bounds():
    bounds = {}
    with open(BOUNDS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            name = r["mapname"].replace("satellite", "").replace(".tif", "")
            bounds[name] = {
                "lat_min": float(r["RB_lat_map"]), "lat_max": float(r["LT_lat_map"]),
                "lng_min": float(r["LT_lon_map"]), "lng_max": float(r["RB_lon_map"]),
            }
    return bounds


def random_point_in_bounds(b, rng):
    lat = rng.uniform(b["lat_min"], b["lat_max"])
    lng = rng.uniform(b["lng_min"], b["lng_max"])
    return lat, lng


def analyze_flight(flight, bounds, rng):
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    tel_path = CSV_DIR / f"video_telemetry_orb_uavvisloc_uavvisloc_{flight}.csv"
    gt = load_ground_truth(gt_path)
    tel = {r["frame"]: r for r in load_telemetry(tel_path)}
    b = bounds[flight]

    diag_m = haversine_flat_m(b["lat_min"], b["lng_min"], b["lat_max"], b["lng_max"])

    actual_errs, null_errs = [], []
    gt_pts, raw_pts = [], []
    for pt in gt:
        row = tel.get(pt["frame"])
        if row is None:
            continue
        err = haversine_flat_m(pt["lat"], pt["lng"], row["raw_lat"], row["raw_lng"])
        actual_errs.append(err)
        gt_pts.append((pt["lat"], pt["lng"]))
        raw_pts.append((row["raw_lat"], row["raw_lng"]))
        for _ in range(NULL_SAMPLES_PER_FRAME):
            rlat, rlng = random_point_in_bounds(b, rng)
            null_errs.append(haversine_flat_m(pt["lat"], pt["lng"], rlat, rlng))

    actual_errs = np.array(actual_errs)
    null_errs = np.array(null_errs)
    ks_stat, ks_p = stats.ks_2samp(actual_errs, null_errs)

    result = {
        "flight": flight,
        "n": len(actual_errs),
        "map_diag_m": diag_m,
        "actual_mean": actual_errs.mean(), "actual_median": np.median(actual_errs),
        "null_mean": null_errs.mean(), "null_median": np.median(null_errs),
        "actual_near_frac": float(np.mean(actual_errs <= NEAR_TRUTH_THRESHOLD_M)),
        "null_near_frac": float(np.mean(null_errs <= NEAR_TRUTH_THRESHOLD_M)),
        "ks_stat": ks_stat, "ks_p": ks_p,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(actual_errs.max(), np.percentile(null_errs, 99)), 60)
    ax.hist(null_errs, bins=bins, alpha=0.45, density=True, label="null (random map guess)", color="#888")
    ax.hist(actual_errs, bins=bins, alpha=0.65, density=True, label="actual (ORB raw)", color="#d62728")
    ax.axvline(NEAR_TRUTH_THRESHOLD_M, color="k", linestyle="--", linewidth=1,
               label=f"{NEAR_TRUTH_THRESHOLD_M:.0f}m near-truth threshold")
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Density")
    ax.set_title(f"Flight {flight}: raw ORB error vs. random-guess null (n={len(actual_errs)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"hist_flight{flight}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    gt_arr = np.array(gt_pts)
    raw_arr = np.array(raw_pts)
    errs_norm = np.clip(actual_errs / diag_m, 0, 1)
    ax.plot(gt_arr[:, 1], gt_arr[:, 0], "-", color="#1f77b4", linewidth=1.5,
            label="ground truth trajectory", zorder=2)
    sc = ax.scatter(raw_arr[:, 1], raw_arr[:, 0], c=errs_norm, cmap="autumn_r", s=14,
                     alpha=0.8, zorder=3, label="ORB raw guess (color = error/map diag)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Flight {flight}: raw guesses vs. true trajectory")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="normalized error")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"map_flight{flight}.png", dpi=130)
    plt.close(fig)

    return result


def main():
    bounds = load_bounds()
    rng = random.Random(RNG_SEED)
    results = [analyze_flight(f, bounds, rng) for f in FLIGHTS]

    print(f"{'Flight':>6} {'n':>5} {'MapDiag':>9} {'ActMean':>9} {'ActMed':>8} "
          f"{'NullMean':>9} {'NullMed':>8} {'Act<500m':>9} {'Null<500m':>10} {'KS_p':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['flight']:>6} {r['n']:>5} {r['map_diag_m']:>8.0f}m "
              f"{r['actual_mean']:>8.0f}m {r['actual_median']:>7.0f}m "
              f"{r['null_mean']:>8.0f}m {r['null_median']:>7.0f}m "
              f"{r['actual_near_frac']:>8.1%} {r['null_near_frac']:>9.1%} "
              f"{r['ks_p']:>10.2e}")

    all_actual_near = sum(r["actual_near_frac"] * r["n"] for r in results) / sum(r["n"] for r in results)
    all_null_near = sum(r["null_near_frac"] * r["n"] for r in results) / sum(r["n"] for r in results)
    print("-" * 100)
    print(f"Combined near-truth (<={NEAR_TRUTH_THRESHOLD_M:.0f}m) fraction: "
          f"actual={all_actual_near:.1%}  null={all_null_near:.1%}")
    print(f"\nPlots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
