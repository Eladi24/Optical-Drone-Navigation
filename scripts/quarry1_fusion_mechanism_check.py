#!/usr/bin/env python3
"""
STRATEGY.md Phase 4, Step 3 -- MECHANISM CHECK ONLY, NOT A PHASE 4 ACCURACY RESULT.

Step 3 is "closed-form MAP estimate fusing odometry displacements with retrieval
anchors, clamping detected outliers" (batch_trajectory_fusion.py's map_fuse_trajectory()).
No dataset on disk has continuous-video odometry AND wide-area retrieval anchors
together: MUN-FRL quarry1 (real Kimera-VIO odometry, Phase 3) has a whole real
flight path that fits inside a ~150x30m box -- far smaller than one satellite
reference-crop, so retrieval there would be testing nothing; UAV-VisLoc (real
retrieval anchors, Phases 0/2/4) has no continuous video or dense telemetry --
consecutive stills are wide-baseline (tens to hundreds of metres, seconds-to-tens-
of-seconds apart), not representative of what a deployed continuous-video system's
odometry would look like. Reached with the user directly after they (correctly)
pushed back on conflating UAV-VisLoc stills with real odometry -- see CLAUDE.md's
"Investigation Log: Phase 4 Step 3" for the full split-validation rationale.

SCOPE (agreed, "split validation"): validate ONLY the fusion algorithm's
MECHANICS -- does it correctly down-weight/reject a real, already-characterized
bad odometry segment when better position information is available nearby --
using quarry1's real Kimera-VIO trajectory (which has a real, non-synthetic
failure: accelerometer bias frozen t=109.2-191.0s, hard reset at t~191.7s, see
CLAUDE.md's Kimera-VIO investigation entries) plus SYNTHESIZED sparse
retrieval-anchor-like corrections (quarry1 has no real retrieval problem to draw
anchors from). This script and map_fuse_trajectory() are purely additive: they do
not touch UAV-VisLoc, batch_trajectory_fusion.py's FLIGHTS/main()/run_flight()
path, or any already-recorded Phase 4 Step 1/2 result.

No VIO-to-world alignment step is applied, and this was verified directly, not
assumed: raw traj_vio.csv x/y already agrees with real PPK ground truth to
within ~0.46m mean during the hover period (elapsed <80s, before any real
motion) -- confirming quarry1's Kimera-VIO was initialized directly in the same
local-ENU frame PPK uses (its first row matches leica0's first ground-truth row
exactly), not a separately-scaled/rotated VIO-internal frame. An earlier version
of this script fit a global weighted similarity (scale+rotation) transform
before this was checked, reasoning from CLAUDE.md's already-documented ~36.5x
scale factor -- that factor turned out to be a SYMPTOM of the same growing
bias-integration drift this check exists to test (position error grows
smoothly from ~0 during hover to >1000m by the reset), not a fixed
frame-calibration bug, so fitting a *global* similarity transform against it
was both unnecessary and actively harmful: with the bad segment covering
nearly half the evaluated window and dominating the fit's variance, the IRLS
alignment converged to a nonsense transform (scale=0.0016, 96.6deg rotation)
that corrupted the whole comparison. Caught by checking the fitted transform's
plausibility directly before trusting it -- the fix was to remove the step
entirely, not to constrain it further. Raw consecutive VIO displacements are
used as odometry input completely unmodified.

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/quarry1_fusion_mechanism_check.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from batch_trajectory_fusion import map_fuse_trajectory
from evaluate_ground_truth import error_stats
from extract_munfrl_bag import load_ppk, project_xy

REPO = Path(__file__).parent.parent
VIO_PATH = REPO / "Datasets" / "MUN_FRL_quarry1" / "kimera_output_baseline_orig" / "traj_vio.csv"
PPK_PATH = REPO / "Datasets" / "MUN_FRL_quarry1" / "ppk_data" / "quarry_1_ppk.pos"
OUT_PATH = REPO / "Images" / "batch_fusion" / "quarry1_mechanism_check.png"

# Elapsed-time-zero used throughout this project's quarry1 analysis (earliest
# corrected cam0/imu0 timestamp -- see scripts/extract_munfrl_bag.py and
# CLAUDE.md's Kimera-VIO investigation entries). Reused verbatim, not re-derived,
# so this script's "t=109.2-191.0s" lines up exactly with what's already
# documented rather than silently drifting onto a different axis.
VIO_T0 = 1645557094.141691

# The real, already-diagnosed bad-odometry span (accelerometer bias frozen,
# CLAUDE.md's "Investigation Log: Kimera-VIO Integration and First Drift
# Measurement on MUN-FRL quarry1" and the two same-day follow-ups).
BAD_WINDOW = (109.2, 191.0)

# Anchor-synthesis parameters -- see module docstring and the approved plan
# (.claude/plans/enumerated-yawning-nygaard.md) for justification of each.
ANCHOR_SPACING_S = 15.0          # untuned midpoint of an instructed 10-20s range
ANCHOR_GOOD_SIGMA_M = 15.0       # roughly this project's own A@20m target band
ANCHOR_OUTLIER_PROB = 0.2        # untuned default
ANCHOR_OUTLIER_MIN_M = 300.0     # matched to this project's own measured raw
ANCHOR_OUTLIER_MAX_M = 3500.0    # wrong-match error magnitudes (CLAUDE.md: raw
                                  # mean 2200-3900m across flights 01/03/04/08/10)
ANCHOR_SEED = 42

MAP_SIGMA_ANCHOR_M = ANCHOR_GOOD_SIGMA_M  # matches the synthesized "good" anchor noise
HOVER_CHECK_ELAPSED_S = 80.0  # pre-motion window used only to verify no alignment is needed


def load_vio_traj(path):
    ts_ns, x, y = [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            ts_ns.append(int(row[0])); x.append(float(row[1])); y.append(float(row[2]))
    order = np.argsort(ts_ns)
    return np.array(ts_ns)[order], np.array(x)[order], np.array(y)[order]


def synthesize_anchors(ppk_t, ppk_xy, vio_t, rng):
    """Sparse, occasionally-corrupted position anchors sampled from real PPK
    ground truth, standing in for real retrieval anchors -- see module docstring
    for parameter justification. Snapped to the nearest VIO keyframe. Anchor
    placement is independent of BAD_WINDOW -- a real retrieval failure has no
    reason to correlate with a VIO failure, and that independence is what makes
    this an honest test rather than a rigged one."""
    t_min, t_max = ppk_t.min(), ppk_t.max()
    anchor_times = np.arange(t_min, t_max, ANCHOR_SPACING_S)
    ax = np.interp(anchor_times, ppk_t, ppk_xy[:, 0])
    ay = np.interp(anchor_times, ppk_t, ppk_xy[:, 1])

    is_outlier = rng.random(len(anchor_times)) < ANCHOR_OUTLIER_PROB
    noisy_x, noisy_y = ax.copy(), ay.copy()

    clean = ~is_outlier
    noisy_x[clean] += rng.normal(0, ANCHOR_GOOD_SIGMA_M, clean.sum())
    noisy_y[clean] += rng.normal(0, ANCHOR_GOOD_SIGMA_M, clean.sum())

    n_out = int(is_outlier.sum())
    mag = rng.uniform(ANCHOR_OUTLIER_MIN_M, ANCHOR_OUTLIER_MAX_M, n_out)
    ang = rng.uniform(0, 2 * np.pi, n_out)
    noisy_x[is_outlier] += mag * np.cos(ang)
    noisy_y[is_outlier] += mag * np.sin(ang)

    anchor_idx = np.array([np.argmin(np.abs(vio_t - at)) for at in anchor_times])
    return anchor_idx, noisy_x, noisy_y, is_outlier


def print_row(label, errs):
    s = error_stats(list(errs))
    a = s["a_at"]
    print(f"{label:>28} {s['mean']:8.1f}m {s['median']:8.1f}m {s['max']:8.1f}m "
          f"{a[10]:6.1%} {a[20]:6.1%}  (n={len(errs)})")


def main():
    print("=" * 100)
    print("Phase 4 Step 3 MECHANISM CHECK (quarry1) -- NOT a Phase 4 accuracy result.")
    print("Real Kimera-VIO odometry + SYNTHESIZED retrieval-anchor-like corrections.")
    print("=" * 100)

    vio_ts_ns, vio_x_raw, vio_y_raw = load_vio_traj(VIO_PATH)
    vio_t = vio_ts_ns / 1e9  # absolute epoch seconds, same axis as ppk_t

    ppk_t, ppk_lat, ppk_lng, _ = load_ppk(PPK_PATH)
    lat0, lng0 = ppk_lat.mean(), ppk_lng.mean()
    ppk_x, ppk_y = project_xy(ppk_lat, ppk_lng, lat0, lng0)
    ppk_xy = np.stack([ppk_x, ppk_y], axis=1)

    # Restrict to VIO keyframes within PPK's real coverage -- the fusion test
    # only makes sense where real ground truth exists to synthesize anchors
    # from and to evaluate against.
    mask = (vio_t >= ppk_t.min()) & (vio_t <= ppk_t.max())
    vio_t, vio_x_raw, vio_y_raw = vio_t[mask], vio_x_raw[mask], vio_y_raw[mask]
    N = len(vio_t)
    elapsed = vio_t - VIO_T0
    print(f"\n{N} VIO keyframes within PPK coverage "
          f"(elapsed {elapsed[0]:.1f}s to {elapsed[-1]:.1f}s)")

    # Sanity check (not a fit): confirm raw VIO already shares PPK's frame/scale
    # during the pre-motion hover period, so no alignment step is warranted --
    # see module docstring for why an earlier version's global similarity fit
    # was wrong and was removed rather than constrained.
    hover = elapsed < HOVER_CHECK_ELAPSED_S
    truth_hx = np.interp(vio_t[hover], ppk_t, ppk_xy[:, 0])
    truth_hy = np.interp(vio_t[hover], ppk_t, ppk_xy[:, 1])
    hover_resid = np.hypot(vio_x_raw[hover] - truth_hx, vio_y_raw[hover] - truth_hy)
    print(f"Hover-period VIO-vs-PPK residual (elapsed <{HOVER_CHECK_ELAPSED_S:.0f}s, "
          f"n={hover.sum()}): mean={hover_resid.mean():.2f}m max={hover_resid.max():.2f}m "
          f"-- confirms no alignment needed")

    # Raw consecutive VIO displacements, completely unmodified.
    odo_dx, odo_dy = np.diff(vio_x_raw), np.diff(vio_y_raw)

    # Raw-VIO-integration-alone baseline: cumulative sum of the aligned
    # displacements, anchored at real ground truth only at the very first
    # keyframe (so the comparison isolates accumulated ODOMETRY error, not a
    # residual initial-position offset).
    truth_x0 = np.interp(vio_t[0], ppk_t, ppk_xy[:, 0])
    truth_y0 = np.interp(vio_t[0], ppk_t, ppk_xy[:, 1])
    baseline_x = truth_x0 + np.concatenate([[0.0], np.cumsum(odo_dx)])
    baseline_y = truth_y0 + np.concatenate([[0.0], np.cumsum(odo_dy)])

    rng = np.random.default_rng(ANCHOR_SEED)
    anchor_idx, anchor_x, anchor_y, is_outlier = synthesize_anchors(ppk_t, ppk_xy, vio_t, rng)
    print(f"{len(anchor_idx)} synthetic anchors "
          f"({is_outlier.sum()} outliers, {(~is_outlier).sum()} clean)")

    fx, fy, w_odo, w_anchor = map_fuse_trajectory(
        vio_t, odo_dx, odo_dy, anchor_idx, anchor_x, anchor_y,
        sigma_anchor=MAP_SIGMA_ANCHOR_M)

    truth_x = np.interp(vio_t, ppk_t, ppk_xy[:, 0])
    truth_y = np.interp(vio_t, ppk_t, ppk_xy[:, 1])
    raw_err = np.hypot(baseline_x - truth_x, baseline_y - truth_y)
    fused_err = np.hypot(fx - truth_x, fy - truth_y)

    in_bad = (elapsed >= BAD_WINDOW[0]) & (elapsed <= BAD_WINDOW[1])
    print(f"\n{'':>28} {'mean':>9} {'median':>9} {'max':>9} {'A@10m':>6} {'A@20m':>6}")
    print_row("Raw VIO integration (all)", raw_err)
    print_row("Fused (all)", fused_err)
    print(f"\n-- split by bad window (t={BAD_WINDOW[0]:.1f}-{BAD_WINDOW[1]:.1f}s, "
          f"n_in_window={in_bad.sum()}) --")
    print_row("Raw, INSIDE bad window", raw_err[in_bad])
    print_row("Fused, INSIDE bad window", fused_err[in_bad])
    print_row("Raw, outside bad window", raw_err[~in_bad])
    print_row("Fused, outside bad window", fused_err[~in_bad])

    # Mechanism check A: does IRLS down-weight odometry factors overlapping the
    # known-bad window vs. elsewhere?
    interval_mid = elapsed[:-1] + np.diff(elapsed) / 2.0
    odo_in_bad = (interval_mid >= BAD_WINDOW[0]) & (interval_mid <= BAD_WINDOW[1])
    print(f"\nMechanism check A (odometry) -- final w_odo, bad window vs. elsewhere:")
    print(f"  inside bad window  (n={odo_in_bad.sum()}):  mean w_odo = {w_odo[odo_in_bad].mean():.3f}")
    print(f"  outside bad window (n={(~odo_in_bad).sum()}): mean w_odo = {w_odo[~odo_in_bad].mean():.3f}")

    # Mechanism check B: does IRLS down-weight synthetically-labeled outlier
    # anchors vs. clean ones? (label known exactly, since we generated it)
    print(f"\nMechanism check B (anchors) -- final w_anchor, outlier vs. clean:")
    print(f"  outlier anchors (n={is_outlier.sum()}): mean w_anchor = {w_anchor[is_outlier].mean():.3f}")
    print(f"  clean anchors   (n={(~is_outlier).sum()}): mean w_anchor = {w_anchor[~is_outlier].mean():.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 11))

    ax1.plot(truth_x, truth_y, "-", color="#1f77b4", linewidth=1.5, label="PPK ground truth", zorder=2)
    ax1.plot(baseline_x, baseline_y, "--", color="#888888", linewidth=1.2,
              label="raw VIO integration (aligned, no fusion)", zorder=3)
    ax1.plot(baseline_x[in_bad], baseline_y[in_bad], "--", color="#000000", linewidth=1.8,
              label="raw VIO, inside bad window", zorder=4)
    ax1.plot(fx, fy, "-", color="#d62728", linewidth=2.0, label="Step 3 fused (this check)", zorder=5)
    sizes = 15 + 60 * (w_anchor / max(w_anchor.max(), 1e-9))
    ax1.scatter(anchor_x[~is_outlier], anchor_y[~is_outlier], s=sizes[~is_outlier],
                 c="#2ca02c", alpha=0.7, marker="o", label="synthetic anchor (clean)", zorder=6)
    ax1.scatter(anchor_x[is_outlier], anchor_y[is_outlier], s=sizes[is_outlier],
                 c="#ff7f0e", alpha=0.7, marker="^", label="synthetic anchor (outlier)", zorder=6)
    ax1.set_xlabel("local x (m)"); ax1.set_ylabel("local y (m)")
    ax1.set_title("quarry1 mechanism check: XY trajectory (MECHANISM CHECK ONLY)")
    ax1.legend(loc="best", fontsize=7)
    ax1.set_aspect("equal", adjustable="datalim")

    ax2.axvspan(BAD_WINDOW[0], BAD_WINDOW[1], color="grey", alpha=0.25, label="known bad window")
    ax2.plot(elapsed, raw_err, "--", color="#888888", linewidth=1.2, label="raw VIO integration error")
    ax2.plot(elapsed, fused_err, "-", color="#d62728", linewidth=1.6, label="Step 3 fused error")
    ax2.set_xlabel("elapsed time (s)"); ax2.set_ylabel("position error vs. PPK (m)")
    ax2.set_title("position error vs. elapsed time")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("STRATEGY.md Phase 4 Step 3 -- MECHANISM CHECK, NOT an accuracy result", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130)
    plt.close(fig)
    print(f"\nPlot written to {OUT_PATH}")


if __name__ == "__main__":
    main()
