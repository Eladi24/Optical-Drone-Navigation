#!/usr/bin/env python3
"""
Compares a navigation telemetry CSV against a hand-annotated ground-truth
CSV (produced by the GroundTruth tool) and reports position error in metres
at each annotated landmark frame.

This is the evaluation harness referenced in the project review: without it
there is no way to tell whether a given estimator/config is actually more
accurate, only whether it "looks smoother."

Usage:
    python3 scripts/evaluate_ground_truth.py \
        --gt "CSV Files/ground_truth_sample1.csv" \
        --telemetry "CSV Files/video_telemetry_hybrid_haifa_north_sample1.csv"

Distance is computed with a flat-earth approximation (matches the C++ code's
own CoordinateUtils::calculateDistance), which is accurate to well under 1%
over the few-hundred-metre spans this project deals with.
"""
import argparse
import csv
import math
import sys

M_PER_DEG_LAT = 111320.0


def m_per_deg_lng(lat_deg: float) -> float:
    return 111320.0 * math.cos(math.radians(lat_deg))


def haversine_flat_m(lat1, lng1, lat2, lng2):
    dlat = (lat2 - lat1) * M_PER_DEG_LAT
    dlng = (lng2 - lng1) * m_per_deg_lng((lat1 + lat2) / 2.0)
    return math.hypot(dlat, dlng)


def load_ground_truth(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "frame": int(r["Frame"]),
                "time_sec": float(r["Time_Sec"]),
                "lat": float(r["Lat"]),
                "lng": float(r["Lng"]),
                "label": r.get("Label", ""),
            })
    return sorted(rows, key=lambda r: r["frame"])


def load_telemetry(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "frame": int(r["Frame"]),
                "raw_lat": float(r["Raw_Lat"]),
                "raw_lng": float(r["Raw_Lng"]),
                "filt_lat": float(r["Filtered_Lat"]),
                "filt_lng": float(r["Filtered_Lng"]),
                "confidence": float(r["Match_Confidence"]),
            })
    return sorted(rows, key=lambda r: r["frame"])


def nearest_telemetry_row(telemetry, frame_idx):
    return min(telemetry, key=lambda r: abs(r["frame"] - frame_idx))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="Ground-truth CSV path")
    ap.add_argument("--telemetry", required=True, help="Navigation telemetry CSV path")
    ap.add_argument("--max-frame-gap", type=int, default=15,
                     help="Warn if nearest telemetry row is more than this many frames away")
    args = ap.parse_args()

    gt = load_ground_truth(args.gt)
    telemetry = load_telemetry(args.telemetry)

    if not gt:
        print(f"No ground-truth points in {args.gt}", file=sys.stderr)
        sys.exit(1)
    if not telemetry:
        print(f"No telemetry rows in {args.telemetry}", file=sys.stderr)
        sys.exit(1)

    raw_errors, filt_errors = [], []
    skipped_gap = 0
    print(f"{'Frame':>6} {'Time(s)':>8} {'Label':>14} {'RawErr(m)':>10} {'FiltErr(m)':>11} {'Conf':>6}")
    print("-" * 62)

    for pt in gt:
        row = nearest_telemetry_row(telemetry, pt["frame"])
        gap = abs(row["frame"] - pt["frame"])
        raw_err = haversine_flat_m(pt["lat"], pt["lng"], row["raw_lat"], row["raw_lng"])
        filt_err = haversine_flat_m(pt["lat"], pt["lng"], row["filt_lat"], row["filt_lng"])

        gap_flag = f"  [gap={gap}f]" if gap > args.max_frame_gap else ""
        print(f"{pt['frame']:>6} {pt['time_sec']:>8.1f} {pt['label']:>14} "
              f"{raw_err:>10.1f} {filt_err:>11.1f} {row['confidence']:>6.2f}{gap_flag}")

        # Points whose nearest telemetry row is beyond max_frame_gap aren't a
        # real comparison (e.g. a partial/killed run whose telemetry stops
        # early -- every later ground-truth point would otherwise silently
        # pair with that same last stale row). Shown above for visibility,
        # excluded here so a partial run's summary reflects only the frames
        # actually processed, not stale carry-forward matches.
        if gap > args.max_frame_gap:
            skipped_gap += 1
            continue
        raw_errors.append(raw_err)
        filt_errors.append(filt_err)

    def stats(vals):
        vals = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        return mean, median, max(vals)

    print("-" * 62)
    if raw_errors:
        raw_mean, raw_median, raw_max = stats(raw_errors)
        filt_mean, filt_median, filt_max = stats(filt_errors)
        print(f"{'RAW':>29}   mean={raw_mean:7.1f}m  median={raw_median:7.1f}m  max={raw_max:7.1f}m")
        print(f"{'FILTERED':>29}   mean={filt_mean:7.1f}m  median={filt_median:7.1f}m  max={filt_max:7.1f}m")
    else:
        print("No ground-truth points within --max-frame-gap of any telemetry row.")
    print(f"\n{len(raw_errors)} ground-truth point(s) evaluated"
          f"{f' ({skipped_gap} skipped, nearest telemetry row > {args.max_frame_gap} frames away)' if skipped_gap else ''}.")


if __name__ == "__main__":
    main()
