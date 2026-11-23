#!/usr/bin/env python3
"""
Per-flight along-track offset between position estimates and GPS ground truth (2026-11-23).

Coarse-to-fine on flights 01/03 showed its residual has a consistent ALONG-TRACK component (03: +14 m,
01: about -7 m) -- the signature of a timing offset between the GPS fix and the camera exposure (the GT is then
offset by speed x delay along the flight line). This script estimates that offset on every flight from the
main pipeline's per-frame anchors (video_telemetry_<estimator>_uavvisloc_<f>.csv, Raw_Lat/Lng), which are noisier
than c2f but number in the hundreds per flight. No new runs.

Anchors used: Measurement_Valid, Inliers >= --min-inliers, within --gate-m of GT (drops wrong-crop picks;
the gate is symmetric so it does not bias the along/cross means much), GT speed >= --min-speed (drops
turnarounds and capture pauses). The residual is split into along-track (GT velocity direction) and
cross-track components; median + bootstrap 95% CI reported. Delay = along-track median / median speed.

    python3 -I scripts/gps_exposure_offset_check.py [--estimator split_vlad_xfeat]
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV_DIR = REPO / "CSV Files"
FLIGHTS = ["01", "03", "04", "08", "10", "02", "05", "06", "11"]
SPLIT = {"01": "dev", "03": "dev", "04": "val", "08": "val", "10": "val",
         "02": "held-out", "05": "held-out", "06": "held-out", "11": "held-out"}
M_LAT = 111320.0


def read_csv(path):
    return list(csv.DictReader(open(path, newline="")))


def boot_ci(x, rng, n=2000):
    if len(x) < 5:
        return (float("nan"), float("nan"))
    meds = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return tuple(np.percentile(meds, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="split_vlad_xfeat")
    ap.add_argument("--min-inliers", type=int, default=16)
    ap.add_argument("--gate-m", type=float, default=60.0)
    ap.add_argument("--min-speed", type=float, default=5.0)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"anchors: {args.estimator}, inliers >= {args.min_inliers}, within {args.gate_m:.0f} m of GT, "
          f"GT speed >= {args.min_speed} m/s\n")
    print(f"{'flight':8s}{'split':10s}{'n':>5s}{'speed':>7s}  {'along med [95% CI]':>26s}  {'cross med [95% CI]':>26s}"
          f"{'delay':>9s}{'|err| med':>11s}")
    for f in FLIGHTS:
        tel = {int(r["Frame"]): r for r in read_csv(CSV_DIR / f"telemetry_uavvisloc_{f}.csv")}
        anchors = read_csv(CSV_DIR / f"video_telemetry_{args.estimator}_uavvisloc_uavvisloc_{f}.csv")
        along, cross, speeds, errs = [], [], [], []
        for a in anchors:
            i = int(a["Frame"])
            if a["Measurement_Valid"] != "1" or int(a["Inliers"]) < args.min_inliers:
                continue
            if i not in tel or i - 1 not in tel or i + 1 not in tel:
                continue
            g, gp, gn = tel[i], tel[i - 1], tel[i + 1]
            lat0 = float(g["Lat"])
            kx = M_LAT * math.cos(math.radians(lat0))
            dt = float(gn["Time_Sec"]) - float(gp["Time_Sec"])
            if dt <= 0:
                continue
            vel = np.array([(float(gn["Lng"]) - float(gp["Lng"])) * kx, (float(gn["Lat"]) - float(gp["Lat"])) * M_LAT]) / dt
            speed = np.linalg.norm(vel)
            if speed < args.min_speed:
                continue
            r = np.array([(float(a["Raw_Lng"]) - float(g["Lng"])) * kx, (float(a["Raw_Lat"]) - lat0) * M_LAT])
            err = np.linalg.norm(r)
            if err > args.gate_m:
                continue
            u = vel / speed
            along.append(float(r @ u))
            cross.append(float(r[0] * u[1] - r[1] * u[0]))   # + = right of track
            speeds.append(speed)
            errs.append(err)
        if not along:
            print(f"{f:8s}{SPLIT[f]:10s}{0:5d}  (no usable anchors)")
            continue
        along, cross = np.array(along), np.array(cross)
        a_med, c_med, v_med = np.median(along), np.median(cross), np.median(speeds)
        alo, ahi = boot_ci(along, rng)
        clo, chi = boot_ci(cross, rng)
        print(f"{f:8s}{SPLIT[f]:10s}{len(along):5d}{v_med:7.1f}  {a_med:+7.1f} [{alo:+6.1f},{ahi:+6.1f}] m     "
              f"{c_med:+7.1f} [{clo:+6.1f},{chi:+6.1f}] m{a_med / v_med:+8.2f}s{np.median(errs):10.1f} m")


if __name__ == "__main__":
    main()
