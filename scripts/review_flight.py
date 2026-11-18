#!/usr/bin/env python3
"""
Per-flight review for one finished dataset run: invariant checks (bug hunting) + metrics, plus a
side-by-side against the committed version of the same telemetry CSV (`git show HEAD:...`).

Written 2026-11-18 for the one-flight-at-a-time rerun protocol: after each flight, run this, read
the INVARIANTS block first (any FAIL/WARN = investigate before starting the next flight), then the
metrics.

Scoring reuses evaluate_ground_truth.compute_stats() -- the one place that pairs telemetry with GT.

Usage:
    python3 scripts/review_flight.py --flight uavvisloc_01 --estimator split_vlad_disk \
        [--log path/to/run.log]      # optional: parses the [PROFILE] block (cache/timing invariants)
        [--no-baseline]              # skip the git HEAD comparison
"""
import argparse
import csv
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "evaluate_ground_truth", REPO_ROOT / "scripts" / "evaluate_ground_truth.py")
egt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(egt)

# "Landed near the truth" threshold for a chosen crop: crops are 300 m, grid spacing 200 m, so a
# correct-crop pick has raw error well under ~200 m. Used only for the discrimination diagnostic.
NEAR_M = 200.0
FAR_M = 1000.0


def pct(x):
    return "   n/a" if x is None else f"{100 * x:5.1f}%"


def m(x):
    return "    n/a" if x is None else f"{x:7.0f}"


def median(v):
    v = sorted(v)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def quantile(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def auc(pos, neg):
    """P(score of a random positive > random negative), ties = 0.5. O(n*m) is fine at this size."""
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def summarize(gt_path, tel_path):
    r = egt.compute_stats(str(gt_path), str(tel_path))
    tel = egt.load_telemetry(str(tel_path))
    if not r or not r.get("n"):
        return r, tel, {}
    tel_by_frame = {t["frame"]: t for t in tel}
    used = [row for row in r["rows"] if row["gap"] <= 15]
    raw = [row["raw_err"] for row in used]
    # Discrimination: do inlier counts separate good picks from bad ones? (The stale-cache bug made
    # them pure noise: every frame was verified against frame 0's keypoints.)
    near_inl, far_inl, near_err = [], [], []
    for row in used:
        t = tel_by_frame.get(row["frame"])
        if not t or not t["measurement_valid"] or t["inliers"] is None:
            continue
        if row["raw_err"] <= NEAR_M:
            near_inl.append(t["inliers"])
            near_err.append(row["raw_err"])
        elif row["raw_err"] >= FAR_M:
            far_inl.append(t["inliers"])
    extra = {
        "near_frac": sum(1 for e in raw if e <= NEAR_M) / len(raw),
        "a50": sum(1 for e in raw if e <= 50) / len(raw),
        "a100": sum(1 for e in raw if e <= 100) / len(raw),
        "near_med_err": median(near_err),
        "near_inl_med": median(near_inl), "far_inl_med": median(far_inl),
        "n_near": len(near_inl), "n_far": len(far_inl),
        "inlier_auc": auc(near_inl, far_inl),
    }
    return r, tel, extra


def invariants(flight, gt_path, tel, r, log_text):
    out = []

    def add(level, msg):
        out.append((level, msg))

    gt = egt.load_ground_truth(str(gt_path))

    # 1. GT frame numbering == image filename number - 1 (the 2026-11-17 flight-08 bug).
    bad = []
    for g in gt:
        mnum = re.search(r"_(\d+)\.\w+$", g["label"] or "")
        if mnum and int(mnum.group(1)) - 1 != g["frame"]:
            bad.append(g["frame"])
    add("FAIL" if bad else "ok", f"GT Frame == filename-1 for all {len(gt)} rows"
        + (f" -- {len(bad)} mismatches, first {bad[:5]}" if bad else ""))

    # 2. GT implied speed (catches numbering/precision/geo bugs in the GT itself).
    speeds = []
    for a, b in zip(gt, gt[1:]):
        dt = b["time_sec"] - a["time_sec"]
        if dt > 0:
            speeds.append(egt.haversine_flat_m(a["lat"], a["lng"], b["lat"], b["lng"]) / dt)
    vmax = max(speeds) if speeds else 0
    add("WARN" if vmax > 40 else "ok", f"GT implied speed: median {median(speeds):.1f} m/s, max {vmax:.1f} m/s"
        + (" (>40 m/s: implausible for this platform)" if vmax > 40 else ""))

    # 3. Coverage: telemetry covers the GT frame range, frames contiguous.
    frames = [t["frame"] for t in tel]
    gaps = [b - a for a, b in zip(frames, frames[1:]) if b - a != 1]
    gt_lo, gt_hi = gt[0]["frame"], gt[-1]["frame"]
    covered = frames and frames[0] <= gt_lo and frames[-1] >= gt_hi
    add("ok" if covered and not gaps else "FAIL",
        f"telemetry frames {frames[0]}..{frames[-1]} ({len(frames)} rows) vs GT {gt_lo}..{gt_hi}"
        + (f"; {len(gaps)} non-contiguous steps" if gaps else "")
        + ("" if covered else " -- RUN INCOMPLETE?"))
    if r and r.get("skipped_gap"):
        add("WARN", f"{r['skipped_gap']} GT points had no telemetry within 15 frames")

    # 4. Stale-output fingerprint: consecutive VALID measurements with bit-identical raw positions.
    rep = sum(1 for a, b in zip(tel, tel[1:])
              if a["measurement_valid"] and b["measurement_valid"]
              and a["raw_lat"] == b["raw_lat"] and a["raw_lng"] == b["raw_lng"])
    nvalid_pairs = max(1, sum(1 for a, b in zip(tel, tel[1:]) if a["measurement_valid"] and b["measurement_valid"]))
    frac = rep / nvalid_pairs
    add("WARN" if frac > 0.05 else "ok",
        f"identical consecutive valid raw positions: {rep}/{nvalid_pairs} ({100*frac:.1f}%)"
        + (" -- stale-cache-like repetition" if frac > 0.05 else ""))

    # 5. Validity: 0.0% invalid was itself a symptom of the stale cache; report, don't judge.
    inv = sum(1 for t in tel if not t["measurement_valid"]) / len(tel)
    inl = [t["inliers"] for t in tel if t["inliers"] is not None and t["measurement_valid"]]
    add("info", f"invalid {100*inv:.1f}%; valid-frame inliers median {median(inl)}, "
        f"p10 {quantile(inl, .1)}, p90 {quantile(inl, .9)}, max {max(inl) if inl else None}")

    # 6. Profile block from the run log (cache eviction + timing).
    if log_text:
        prof = dict()
        for line in log_text.splitlines():
            mm = re.match(r"\[PROFILE\] (.+?)\s{2,}([\d.]+)\s+(\d+)\s+([\d.]+)", line)
            if mm:
                prof[mm.group(1).strip()] = (float(mm.group(2)), int(mm.group(3)), float(mm.group(4)))
        ev = prof.get("q beginFrame: stale entry evicted", (0, 0, 0))[1]
        ne = prof.get("q beginFrame: nothing to evict", (0, 0, 0))[1]
        reused = prof.get("z query buffer address: REUSED", (0, 0, 0))[1]
        nframes = prof.get("0 FRAME TOTAL (excl. decode)", (0, 0, 0))[1]
        if nframes:
            okc = (ev + ne == nframes) and (ev == reused)
            add("ok" if okc else "FAIL",
                f"query cache: {ev} evicted + {ne} fresh = {ev+ne} beginFrame calls for {nframes} frames; "
                f"{reused} reused buffer addresses")
            pw = prof.get("p heading pre-warp applied", (0, 0, 0))[1]
            add("ok" if pw == nframes else "FAIL",
                f"heading pre-warp applied on {pw}/{nframes} frames (default-on since 2026-11-18)")
            ft = prof["0 FRAME TOTAL (excl. decode)"][0] / 1000
            mt = prof.get("3b matching, all candidates (wall)", (0, 0, 0))[0] / 1000
            add("info", f"timing: {ft:.2f} s/frame, matching {mt:.2f} s/frame")
        else:
            add("WARN", "no [PROFILE] block in log (run without DRONENAV_PROFILE=1, or crashed)")
        if re.search(r"terminate called|Segmentation|Aborted|what\(\):", log_text):
            add("FAIL", "crash/exception text in run log")
    return out


def print_metrics(label, r, extra):
    if not r or not r.get("n"):
        print(f"  {label:<14} (no data)")
        return
    rec, pdm = r["recall"], r["pdm_at_k"]
    print(f"  {label:<14} n={r['n']:<4} raw med {m(r['raw']['median'])} mean {m(r['raw']['mean'])} | "
          f"filt med {m(r['filt']['median'])} mean {m(r['filt']['mean'])}")
    print(f"  {'':<14} A@20 {pct(r['raw']['a_at'][20])} A@50 {pct(extra['a50'])} A@100 {pct(extra['a100'])} "
          f"A@200 {pct(extra['near_frac'])} | R@1 {pct(rec[1])} R@5 {pct(rec[5])} R@10 {pct(rec[10])} | "
          f"PDM@5 med {m(pdm[5]['median'] if pdm[5] else None)}")
    print(f"  {'':<14} discrimination: inliers on near(<={NEAR_M:.0f}m) picks med {extra['near_inl_med']} "
          f"(n={extra['n_near']}) vs far(>={FAR_M:.0f}m) med {extra['far_inl_med']} (n={extra['n_far']}), "
          f"AUC {extra['inlier_auc'] if extra['inlier_auc'] is None else round(extra['inlier_auc'], 3)}; "
          f"near-pick median error {m(extra['near_med_err'])} m")
    print(f"  {'':<14} invalid {pct(r['invalid_rate'])}  outlier {pct(r['outlier_rate'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", required=True)
    ap.add_argument("--estimator", default="split_vlad_disk")
    ap.add_argument("--log")
    ap.add_argument("--no-baseline", action="store_true")
    a = ap.parse_args()

    gt_path = REPO_ROOT / "CSV Files" / f"ground_truth_{a.flight}.csv"
    rel = f"CSV Files/video_telemetry_{a.estimator}_uavvisloc_{a.flight}.csv"
    tel_path = REPO_ROOT / rel
    for p in (gt_path, tel_path):
        if not p.exists():
            sys.exit(f"missing {p}")
    log_text = Path(a.log).read_text(errors="replace") if a.log else ""

    r, tel, extra = summarize(gt_path, tel_path)
    print(f"\n=== {a.flight} / {a.estimator} ===\nINVARIANTS")
    for level, msg in invariants(a.flight, gt_path, tel, r, log_text):
        print(f"  [{level:>4}] {msg}")

    print("METRICS")
    print_metrics("this run", r, extra)
    if not a.no_baseline:
        try:
            old = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            print("  (no committed version to compare against)")
            return
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as f:
            f.write(old)
        ro, _, eo = summarize(gt_path, f.name)
        print_metrics("git HEAD", ro, eo)
        Path(f.name).unlink()


if __name__ == "__main__":
    main()
