#!/usr/bin/env python3
"""
STRATEGY.md Phase 4 gate: A@20m on held-out UAV-VisLoc flights (target 15-30m
mean absolute error) -- the literal last unstarted Phase 4 task. Every number
recorded elsewhere in this project (Step 1's batch-fusion win, Step 2's negative
result) used dev (01, 03) or validation (04, 08, 10) flights; this is the first
run against flights the fusion hyperparameters (BANDWIDTH_S, TUKEY_C, WINDOW_S,
ROTATION_CAP_DEG, etc., all in batch_trajectory_fusion.py) were never tuned
against.

Held-out set per STRATEGY.md Sec 6.1: {02, 05, 06, 07, 09, 11}. Two flights
excluded here, both discovered during this run (not anticipated at planning
time):

  - 09: 4 separate GeoTIFF tiles, not yet supported by
    convert_uavvisloc_satellite.py -- see include/DatasetSamples.hpp's comment.
  - 07: its own raw <flight>.csv has a different, shorter schema (6 columns:
    num,filename,date,lat,lon,height -- missing Omega,Kappa,Phi1,Phi2, which
    every other flight's telemetry CSV has). TelemetryImporter parsed zero
    rows from it ("no rows parsed", confirmed directly). A real, one-off data
    anomaly in this specific flight's UAV-VisLoc release, not a bug in this
    run or a code regression -- fixing TelemetryImporter's parser to handle a
    variable column count is a real but separate change, out of scope for
    this gate pass (same "defer a real fix that needs new pipeline code"
    call already made for flight 09).

The remaining 4 split into two groups by a directly-verified altitude-spread
check (own per-flight telemetry `height` column, see DatasetSamples.hpp's
comment for the exact numbers):

  CLEAN_FLIGHTS   -- 02: tight altitude spread, AGL-plausible, same standard
                     as every dev/validation flight.
  CAUTION_FLIGHTS -- 05, 06, 11: spread and/or absolute height strongly
                     suggest MSL, not AGL, over mountainous terrain --
                     GSD/footprint math is known-compromised for these three,
                     same category as flight 08's stale-imagery caveat
                     elsewhere in this project. Geo-referencing itself (WHERE
                     the map is centered) was directly sanity-checked and
                     confirmed correct for all 3 (frame-1 crop-vs-photo visual
                     match) -- it's specifically the altitude/GSD assumption
                     that's compromised, not the position. Reported
                     SEPARATELY, never pooled into the clean-flight number,
                     so neither reading misrepresents the other.

This script does not modify batch_trajectory_fusion.py or its FLIGHTS list --
that file's hyperparameters stay exactly as tuned-and-frozen on dev/validation
flights. It imports and reuses run_flight() (prints the existing full
raw/Kalman/batch-fused/Procrustes table and saves a trajectory plot per flight,
unchanged) and separately reproduces the minimal per-flight raw/fused error
arrays (via the same already-existing load_raw_track/project_to_local_xy/
batch_fuse functions) purely to POOL them within each group for a true
combined A@20m -- not an average of per-flight percentages.

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/held_out_gate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from batch_trajectory_fusion import (
    CSV_DIR, batch_fuse, load_raw_track, project_to_local_xy, run_flight,
)
from evaluate_ground_truth import error_stats

CLEAN_FLIGHTS = ["02"]
CAUTION_FLIGHTS = ["05", "06", "11"]


def pooled_errors(flight):
    """Reproduces run_flight()'s raw/fused local-xy error arrays for one
    flight, without touching batch_trajectory_fusion.py -- needed here only
    to pool points across flights within a group (run_flight() itself
    already reports and plots the full per-flight comparison)."""
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    tel_path = CSV_DIR / f"video_telemetry_orb_uavvisloc_uavvisloc_{flight}.csv"
    frames, t, raw_lat, raw_lng, gt_lat, gt_lng = load_raw_track(gt_path, tel_path)
    lat0, lng0 = gt_lat.mean(), gt_lng.mean()
    x, y = project_to_local_xy(raw_lat, raw_lng, lat0, lng0)
    gx, gy = project_to_local_xy(gt_lat, gt_lng, lat0, lng0)
    fx, fy, _, _ = batch_fuse(t, x, y)
    raw_err = np.hypot(x - gx, y - gy)
    fused_err = np.hypot(fx - gx, fy - gy)
    return raw_err, fused_err


def print_a20_row(label, errs):
    s = error_stats(list(errs))
    a = s["a_at"]
    print(f"{label:>28} {s['mean']:8.1f}m {s['median']:8.1f}m {s['max']:8.1f}m "
          f"A@10m={a[10]:6.1%} A@20m={a[20]:6.1%}  (n={len(errs)})")


def report_group(name, flights):
    print(f"\n{'=' * 100}\n{name}: {', '.join(flights)}\n{'=' * 100}")
    raw_all, fused_all = [], []
    for f in flights:
        raw_err, fused_err = pooled_errors(f)
        raw_all.append(raw_err); fused_all.append(fused_err)
        print(f"\n-- flight {f} (n={len(raw_err)}) --")
        print_a20_row("Raw", raw_err)
        print_a20_row("Batch-fused (Step 1)", fused_err)
    raw_all = np.concatenate(raw_all)
    fused_all = np.concatenate(fused_all)
    print(f"\n-- {name} POOLED (n={len(raw_all)}) --")
    print_a20_row("Raw", raw_all)
    print_a20_row("Batch-fused (Step 1)", fused_all)
    return raw_all, fused_all


def main():
    print("STRATEGY.md Phase 4 GATE -- held-out UAV-VisLoc flights (never tuned against).")
    print("Target: A@20m mean absolute error 15-30m.")
    print("Flight 09 deferred (multi-tile, unsupported). Flight 07 excluded (telemetry schema anomaly, 0 rows parsed).")

    print("\n" + "#" * 100)
    print("# Per-flight detail (full raw/Kalman/batch-fused/Procrustes tables + trajectory plots)")
    print("#" * 100)
    for f in CLEAN_FLIGHTS + CAUTION_FLIGHTS:
        run_flight(f)

    clean_raw, clean_fused = report_group("CLEAN (AGL-plausible)", CLEAN_FLIGHTS)
    caution_raw, caution_fused = report_group("CAUTION (known altitude/MSL ambiguity)", CAUTION_FLIGHTS)

    print(f"\n{'=' * 100}\nGATE SUMMARY -- clean and caution groups NEVER pooled together\n{'=' * 100}")
    print_a20_row("Clean, raw", clean_raw)
    print_a20_row("Clean, batch-fused", clean_fused)
    print_a20_row("Caution, raw", caution_raw)
    print_a20_row("Caution, batch-fused", caution_fused)


if __name__ == "__main__":
    main()
