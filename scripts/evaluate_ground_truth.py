#!/usr/bin/env python3
"""
Compares a navigation telemetry CSV against a hand-annotated ground-truth
CSV (produced by the GroundTruth tool, or by TelemetryImporter for a
telemetry-labeled dataset) and reports position error in metres at each
annotated/ground-truthed frame.

This is the evaluation harness referenced in the project review: without it
there is no way to tell whether a given estimator/config is actually more
accurate, only whether it "looks smoother."

STRATEGY.md Phase 0 extended this with A@5/10/20m threshold-accuracy
(fraction of points within N metres -- mean is dominated by catastrophic
outliers and hides whether the typical frame is actually good) and
recall@{1,5,10} (derived from the telemetry's Correct_Candidate_Rank column
-- was the geometrically correct crop even retrieved, independent of
whether the fusion layer trusted it). STRATEGY.md's Phase 2 learned-retrieval
pass added PDM@K (Top{1,5,10}_Min_Dist_M columns): the minimum ground-truth
distance among the top-K score-ranked candidates -- "if a perfect downstream
selector always picked the best of the top-K, what's the position error."
NOTE: this is this project's own labeled INTERPRETATION of AnyVisLoc's
PDM@K, in its stated spirit (recall@K is binary and throws away how close a
near-miss was), not a verified reproduction of their exact published
formula -- see CLAUDE.md. The core comparison logic is exposed as
compute_stats() so scripts/run_benchmark.py can call it directly across
multiple flights without reimplementing the ground-truth/telemetry matching.

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


def _float_or_none(s):
    s = (s or "").strip()
    return float(s) if s else None


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
            rank_s = (r.get("Correct_Candidate_Rank") or "").strip()
            rows.append({
                "frame": int(r["Frame"]),
                "raw_lat": float(r["Raw_Lat"]),
                "raw_lng": float(r["Raw_Lng"]),
                "filt_lat": float(r["Filtered_Lat"]),
                "filt_lng": float(r["Filtered_Lng"]),
                "confidence": float(r["Match_Confidence"]),
                # Columns below added by STRATEGY.md Phase 0; default gracefully
                # (None/0) so this script still works against older telemetry CSVs.
                "outlier_rejected": r.get("Outlier_Rejected") == "1",
                "inliers": int(r["Inliers"]) if "Inliers" in r and r["Inliers"] != "" else None,
                "measurement_valid": (r.get("Measurement_Valid") != "0"
                                       if "Measurement_Valid" in r else True),
                "correct_candidate_rank": int(rank_s) if rank_s else None,
                # PDM@K proxy columns (STRATEGY.md Phase 2) -- blank/absent on
                # older telemetry CSVs, same graceful-default pattern as above.
                "top1_min_dist": _float_or_none(r.get("Top1_Min_Dist_M")),
                "top5_min_dist": _float_or_none(r.get("Top5_Min_Dist_M")),
                "top10_min_dist": _float_or_none(r.get("Top10_Min_Dist_M")),
            })
    return sorted(rows, key=lambda r: r["frame"])


def nearest_telemetry_row(telemetry, frame_idx):
    return min(telemetry, key=lambda r: abs(r["frame"] - frame_idx))


def error_stats(vals):
    """Mean/median/max plus A@5/10/20m threshold-accuracy fractions."""
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    a_at = {t: sum(1 for v in vals if v <= t) / n for t in (5, 10, 20)}
    return {"mean": mean, "median": median, "max": max(vals), "a_at": a_at}


def compute_stats(gt_path, telemetry_path, max_frame_gap=15):
    """Core ground-truth-vs-telemetry comparison, importable for the aggregate
    benchmark runner (scripts/run_benchmark.py) as well as this script's own
    CLI. Returns None if either file has no usable rows; otherwise a dict:

        n, skipped_gap        -- ground-truth points evaluated / skipped
        raw, filt             -- error_stats() dicts for raw/filtered position
        recall                -- {1: frac, 5: frac, 10: frac} or None per-k
                                  if no ranked points exist
        n_recall              -- ground-truth points with a usable rank
        outlier_rate          -- fraction of ALL telemetry frames rejected
                                  as Kalman outliers (not just gt-matched ones)
        invalid_rate          -- fraction of ALL telemetry frames with
                                  measurement_valid=0 (no RANSAC-verified
                                  candidate at all -- distinct from outlier
                                  rejection, see STRATEGY.md Sec 4.3 #2)
        total_frames          -- telemetry row count, for context
        rows                  -- per-point detail (frame/time/label/errors/gap)
    """
    gt = load_ground_truth(gt_path)
    telemetry = load_telemetry(telemetry_path)
    if not gt or not telemetry:
        return None

    raw_errors, filt_errors, ranks = [], [], []
    pdm_dists = {1: [], 5: [], 10: []}
    skipped_gap = 0
    rows = []

    for pt in gt:
        row = nearest_telemetry_row(telemetry, pt["frame"])
        gap = abs(row["frame"] - pt["frame"])
        raw_err = haversine_flat_m(pt["lat"], pt["lng"], row["raw_lat"], row["raw_lng"])
        filt_err = haversine_flat_m(pt["lat"], pt["lng"], row["filt_lat"], row["filt_lng"])
        rows.append({**pt, "gap": gap, "raw_err": raw_err, "filt_err": filt_err,
                     "confidence": row["confidence"], "rank": row["correct_candidate_rank"],
                     "top1_min_dist": row["top1_min_dist"], "top5_min_dist": row["top5_min_dist"],
                     "top10_min_dist": row["top10_min_dist"]})

        # Points whose nearest telemetry row is beyond max_frame_gap aren't a
        # real comparison (e.g. a partial/killed run whose telemetry stops
        # early -- every later ground-truth point would otherwise silently
        # pair with that same last stale row).
        if gap > max_frame_gap:
            skipped_gap += 1
            continue
        raw_errors.append(raw_err)
        filt_errors.append(filt_err)
        if row["correct_candidate_rank"] is not None:
            ranks.append(row["correct_candidate_rank"])
        for k, key in ((1, "top1_min_dist"), (5, "top5_min_dist"), (10, "top10_min_dist")):
            if row[key] is not None:
                pdm_dists[k].append(row[key])

    result = {"n": len(raw_errors), "skipped_gap": skipped_gap, "rows": rows}
    if not raw_errors:
        return result

    result["raw"] = error_stats(raw_errors)
    result["filt"] = error_stats(filt_errors)

    result["n_recall"] = len(ranks)
    result["recall"] = ({k: sum(1 for r in ranks if 0 < r <= k) / len(ranks) for k in (1, 5, 10)}
                         if ranks else {1: None, 5: None, 10: None})

    # PDM@K proxy (see module docstring caveat) -- mean/median of the
    # min-distance-among-top-K values, per K; None for a K with no data
    # (e.g. older telemetry CSVs, or fewer than K candidates every frame).
    def _mean_median(vals):
        if not vals:
            return None
        vals = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        return {"mean": mean, "median": median, "n": n}

    result["pdm_at_k"] = {k: _mean_median(pdm_dists[k]) for k in (1, 5, 10)}

    total_frames = len(telemetry)
    outlier_frames = sum(1 for r in telemetry if r["outlier_rejected"])
    invalid_frames = sum(1 for r in telemetry if not r["measurement_valid"])
    result["total_frames"] = total_frames
    result["outlier_frames"] = outlier_frames
    result["invalid_frames"] = invalid_frames
    result["outlier_rate"] = outlier_frames / total_frames
    result["invalid_rate"] = invalid_frames / total_frames

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="Ground-truth CSV path")
    ap.add_argument("--telemetry", required=True, help="Navigation telemetry CSV path")
    ap.add_argument("--max-frame-gap", type=int, default=15,
                     help="Warn if nearest telemetry row is more than this many frames away")
    args = ap.parse_args()

    result = compute_stats(args.gt, args.telemetry, args.max_frame_gap)
    if result is None:
        if not load_ground_truth(args.gt):
            print(f"No ground-truth points in {args.gt}", file=sys.stderr)
        else:
            print(f"No telemetry rows in {args.telemetry}", file=sys.stderr)
        sys.exit(1)

    print(f"{'Frame':>6} {'Time(s)':>8} {'Label':>14} {'RawErr(m)':>10} {'FiltErr(m)':>11} {'Conf':>6}")
    print("-" * 62)
    for r in result["rows"]:
        gap_flag = f"  [gap={r['gap']}f]" if r["gap"] > args.max_frame_gap else ""
        print(f"{r['frame']:>6} {r['time_sec']:>8.1f} {r['label']:>14} "
              f"{r['raw_err']:>10.1f} {r['filt_err']:>11.1f} {r['confidence']:>6.2f}{gap_flag}")

    print("-" * 62)
    if result["n"]:
        raw, filt = result["raw"], result["filt"]

        def fmt(s):
            a = s["a_at"]
            return (f"mean={s['mean']:7.1f}m  median={s['median']:7.1f}m  max={s['max']:7.1f}m  "
                    f"A@5m={a[5]:5.1%}  A@10m={a[10]:5.1%}  A@20m={a[20]:5.1%}")

        print(f"{'RAW':>10}   {fmt(raw)}")
        print(f"{'FILTERED':>10}   {fmt(filt)}")

        if result["n_recall"]:
            rec = result["recall"]
            fmt_r = lambda v: f"{v:5.1%}" if v is not None else "  n/a"
            print(f"{'RECALL':>10}   @1={fmt_r(rec[1])}  @5={fmt_r(rec[5])}  @10={fmt_r(rec[10])}"
                  f"  (n={result['n_recall']} ranked points)")
        else:
            print(f"{'RECALL':>10}   n/a (telemetry has no Correct_Candidate_Rank data)")

        pdm = result["pdm_at_k"]
        if any(pdm[k] for k in (1, 5, 10)):
            fmt_pdm = lambda d: f"{d['mean']:.1f}m/{d['median']:.1f}m (n={d['n']})" if d else "n/a"
            print(f"{'PDM@K (proxy)':>10}   @1={fmt_pdm(pdm[1])}  @5={fmt_pdm(pdm[5])}  @10={fmt_pdm(pdm[10])}"
                  f"  [mean/median min-dist-in-top-K; this project's own interpretation of AnyVisLoc's"
                  f" PDM@K, not a verified reproduction -- see CLAUDE.md]")
        else:
            print(f"{'PDM@K':>10}   n/a (telemetry has no Top{{1,5,10}}_Min_Dist_M data)")

        print(f"{'OUTLIERS':>10}   {result['outlier_rate']:5.1%} of {result['total_frames']} frames "
              f"rejected  |  {'INVALID':>7} {result['invalid_rate']:5.1%} (no RANSAC-verified candidate)")
    else:
        print("No ground-truth points within --max-frame-gap of any telemetry row.")
    skip_note = ""
    if result["skipped_gap"]:
        skip_note = (f" ({result['skipped_gap']} skipped, nearest telemetry row "
                     f"> {args.max_frame_gap} frames away)")
    print(f"\n{result['n']} ground-truth point(s) evaluated{skip_note}.")


if __name__ == "__main__":
    main()
