#!/usr/bin/env python3
"""
Aggregate benchmark runner for the telemetry-labeled dataset pipeline
(STRATEGY.md Phase 0 gate: "one command, N flights, one comparable table").

For each requested flight, runs `./build/Main <estimator> d <flight>` (the
existing dataset pipeline, filtered to that one flight via the sample_name
substring match added to Main.cpp's argv[3] handling), then evaluates the
resulting telemetry CSV against that flight's ground truth using
scripts/evaluate_ground_truth.py's compute_stats() -- imported directly, not
reimplemented, so there is exactly one place that knows how to compare a
telemetry CSV against ground truth.

Config-driven: which flights and which estimator to run are CLI arguments,
not constants edited in this file.

Usage:
    python3 scripts/run_benchmark.py --flights uavvisloc_01,uavvisloc_03 --estimator orb

    # Re-evaluate already-produced telemetry CSVs without re-running the
    # (slow) C++ pipeline -- useful when iterating on the eval side only:
    python3 scripts/run_benchmark.py --flights uavvisloc_01,uavvisloc_03 --estimator orb --skip-run
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# evaluate_ground_truth.py isn't a package -- load it directly from its path
# so this script can import compute_stats() without needing sys.path tricks
# or a setup.py the rest of this project doesn't have.
_spec = importlib.util.spec_from_file_location(
    "evaluate_ground_truth", REPO_ROOT / "scripts" / "evaluate_ground_truth.py")
evaluate_ground_truth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_ground_truth)


def telemetry_path_for(estimator: str, flight: str) -> Path:
    # Matches VideoProcessing.cpp's naming exactly: video_telemetry_<algo>_<location_name>_<sample_name>.csv
    # location_name is always "uavvisloc" for this pipeline (see DatasetSamples.hpp).
    return REPO_ROOT / "CSV Files" / f"video_telemetry_{estimator}_uavvisloc_{flight}.csv"


def ground_truth_path_for(flight: str) -> Path:
    return REPO_ROOT / "CSV Files" / f"ground_truth_{flight}.csv"


def run_flight(main_binary: str, estimator: str, flight: str) -> None:
    print(f"\n{'=' * 80}\nRunning {flight} ({estimator})\n{'=' * 80}")
    subprocess.run([main_binary, estimator, "d", flight], cwd=REPO_ROOT, check=True)


def fmt_pct(v):
    return f"{v:5.1%}" if v is not None else "  n/a"


def fmt_pdm5(pdm_at_k):
    """PDM@5 median only (STRATEGY.md Sec 7's PDM@K proxy -- see
    evaluate_ground_truth.py's docstring caveat: this project's own
    interpretation, not a verified reproduction of AnyVisLoc's exact
    formula). Full PDM@{1,5,10} mean/median is available per-flight via
    evaluate_ground_truth.py directly; this table stays compact with just
    the one column most directly paired with the existing R@5 column."""
    d = pdm_at_k.get(5) if pdm_at_k else None
    return f"{d['median']:>6.0f}m" if d else "   n/a"


def print_table(rows, combined):
    header = (f"{'Flight':<16} {'n':>4} {'RawMean':>9} {'RawMed':>8} {'FiltMean':>9} {'FiltMed':>8} "
              f"{'A@5m':>6} {'A@10m':>6} {'A@20m':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6} "
              f"{'PDM@5':>7} {'Outlier':>8} {'Invalid':>8}")
    print(header)
    print("-" * len(header))
    for name, r in rows:
        if r is None or not r.get("n"):
            print(f"{name:<16}   (no comparable ground-truth points)")
            continue
        raw, filt, rec = r["raw"], r["filt"], r["recall"]
        print(f"{name:<16} {r['n']:>4} {raw['mean']:>8.1f}m {raw['median']:>7.1f}m "
              f"{filt['mean']:>8.1f}m {filt['median']:>7.1f}m "
              f"{filt['a_at'][5]:>6.1%} {filt['a_at'][10]:>6.1%} {filt['a_at'][20]:>6.1%} "
              f"{fmt_pct(rec[1])} {fmt_pct(rec[5])} {fmt_pct(rec[10])} "
              f"{fmt_pdm5(r.get('pdm_at_k'))} "
              f"{r['outlier_rate']:>7.1%} {r['invalid_rate']:>7.1%}")
    print("-" * len(header))
    if combined:
        raw, filt, rec = combined["raw"], combined["filt"], combined["recall"]
        print(f"{'COMBINED':<16} {combined['n']:>4} {raw['mean']:>8.1f}m {raw['median']:>7.1f}m "
              f"{filt['mean']:>8.1f}m {filt['median']:>7.1f}m "
              f"{filt['a_at'][5]:>6.1%} {filt['a_at'][10]:>6.1%} {filt['a_at'][20]:>6.1%} "
              f"{fmt_pct(rec[1])} {fmt_pct(rec[5])} {fmt_pct(rec[10])} "
              f"{fmt_pdm5(combined.get('pdm_at_k'))} "
              f"{combined['outlier_rate']:>7.1%} {combined['invalid_rate']:>7.1%}")


def combine(results, max_frame_gap):
    """Pools every flight's per-point rows into one set of stats, rather than
    averaging each flight's already-summarized mean/median (which would
    silently misweight flights with different point counts)."""
    pooled_raw, pooled_filt, pooled_ranks = [], [], []
    pooled_pdm = {1: [], 5: [], 10: []}
    total_frames = total_outlier = total_invalid = 0

    for r in results:
        if not r or not r.get("n"):
            continue
        for row in r["rows"]:
            if row["gap"] > max_frame_gap:
                continue
            pooled_raw.append(row["raw_err"])
            pooled_filt.append(row["filt_err"])
            if row["rank"] is not None:
                pooled_ranks.append(row["rank"])
            for k, key in ((1, "top1_min_dist"), (5, "top5_min_dist"), (10, "top10_min_dist")):
                if row.get(key) is not None:
                    pooled_pdm[k].append(row[key])
        total_frames += r["total_frames"]
        total_outlier += r["outlier_frames"]
        total_invalid += r["invalid_frames"]

    if not pooled_raw:
        return None

    recall = ({k: sum(1 for x in pooled_ranks if 0 < x <= k) / len(pooled_ranks) for k in (1, 5, 10)}
              if pooled_ranks else {1: None, 5: None, 10: None})

    def _mean_median(vals):
        if not vals:
            return None
        vals = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        return {"mean": mean, "median": median, "n": n}

    return {
        "n": len(pooled_raw),
        "raw": evaluate_ground_truth.error_stats(pooled_raw),
        "filt": evaluate_ground_truth.error_stats(pooled_filt),
        "recall": recall,
        "pdm_at_k": {k: _mean_median(pooled_pdm[k]) for k in (1, 5, 10)},
        "outlier_rate": total_outlier / total_frames if total_frames else None,
        "invalid_rate": total_invalid / total_frames if total_frames else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", required=True,
                     help="Comma-separated DatasetSampleConfig sample_name values, "
                          "e.g. uavvisloc_01,uavvisloc_03")
    ap.add_argument("--estimator", default="orb",
                     help="Algorithm passed to Main (orb/sift/hybrid/akaze/optical_flow). Default: orb")
    ap.add_argument("--main-binary", default=str(REPO_ROOT / "build" / "Main"),
                     help="Path to the built Main executable")
    ap.add_argument("--max-frame-gap", type=int, default=15,
                     help="Passed through to evaluate_ground_truth.compute_stats()")
    ap.add_argument("--skip-run", action="store_true",
                     help="Evaluate existing telemetry CSVs without re-running Main "
                          "(faster iteration on the eval side only)")
    args = ap.parse_args()

    flights = [f.strip() for f in args.flights.split(",") if f.strip()]
    if not flights:
        print("No flights specified.", file=sys.stderr)
        sys.exit(1)

    rows = []
    results_for_combine = []
    for flight in flights:
        if not args.skip_run:
            run_flight(args.main_binary, args.estimator, flight)

        gt_path = ground_truth_path_for(flight)
        telemetry_path = telemetry_path_for(args.estimator, flight)
        if not gt_path.exists() or not telemetry_path.exists():
            print(f"Missing {gt_path if not gt_path.exists() else telemetry_path} "
                  f"for {flight} -- skipping evaluation.", file=sys.stderr)
            rows.append((flight, None))
            continue

        result = evaluate_ground_truth.compute_stats(str(gt_path), str(telemetry_path), args.max_frame_gap)
        rows.append((flight, result))
        results_for_combine.append(result)

    combined = combine(results_for_combine, args.max_frame_gap)

    print(f"\n{'=' * 80}\nSTRATEGY.md Phase 0 gate -- {args.estimator}, flights: {', '.join(flights)}\n{'=' * 80}")
    print_table(rows, combined)


if __name__ == "__main__":
    main()
