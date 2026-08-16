#!/usr/bin/env python3
"""
STRATEGY.md Phase 4, Step 1 prototype: batch (non-causal, whole-trajectory) robust
trajectory fusion, evaluated offline over already-logged ORB telemetry. See the
approved plan at .claude/plans/reactive-discovering-orbit.md for full design
rationale -- summarized here:

NaviLoc's central claim (STRATEGY.md Sec 3): per-frame matching against the
cross-domain satellite/drone gap is irreducibly unreliable, but incorrect matches
scatter across the map while correct matches cluster near the true trajectory. A
robust, whole-sequence estimator can exploit that even when most individual frames
are wrong -- something no causal filter (DroneKalmanFilter, ParticleFilter -- both
already tried in this project, both plateaued) can do, since a causal filter can
never use future frames to judge a past one.

This is this project's own interpretation of STRATEGY.md's "global SE(2) alignment"
task, NOT a literal implementation of it -- the literal wording assumes Phase 3
odometry (relative-motion) data this project doesn't have yet (Phase 3 is blocked
on dataset acquisition, see CLAUDE.md). In its place, this uses a weaker, IMU-free
stand-in: a soft "bounded-speed, smooth-path" prior, applied non-causally (whole
trajectory at once) instead of causally -- the same category of assumption
DroneKalmanFilter's own constant-velocity model already makes, just not restricted
to only seeing the past.

METHOD: robust local-linear regression (a "robust LOESS") via IRLS with Tukey's
biweight loss, fit independently to x(t) and y(t) (local planar projection of
Raw_Lat/Raw_Lng vs. real Time_Sec). Each iteration re-weights every frame by how
far its raw match sits from the current fit; frames whose raw match keeps agreeing
with a locally-consistent majority end up with high weight, frames that don't
(false-attractor matches, scattered wrong crops) get zeroed out. This directly
implements "no single match needs to be correct, only for correct matches to
cluster more tightly than incorrect ones."

Deliberately offline/Python-only and using only the single best-match per frame
(Raw_Lat/Raw_Lng, already logged) -- no VideoProcessing.cpp changes, no new
IPositionEstimator, no live pipeline integration. Mirrors how this project already
prototyped the same-domain experiment and the clustering-assumption check as
standalone diagnostics before (if ever) wiring them into the live pipeline.

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/batch_trajectory_fusion.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_ground_truth import (
    compute_stats, error_stats, load_ground_truth, load_telemetry,
    m_per_deg_lng, M_PER_DEG_LAT, nearest_telemetry_row,
)

REPO = Path(__file__).parent.parent
CSV_DIR = REPO / "CSV Files"
OUT_DIR = REPO / "Images" / "batch_fusion"

# Dev flights (01, 03) first validated these hyperparameters; validation flights
# (04, 08, 10) added afterward to check generalization -- run with UNCHANGED
# hyperparameters below, not retuned per flight, per this project's dev/validation
# discipline (STRATEGY.md Sec 6.1). Held-out flights (02/05/06/07/09/11) deliberately
# not touched -- reserved for Phase 4's actual gate, not a Step-1 prototype.
FLIGHTS = ["01", "03", "04", "08", "10"]
BANDWIDTH_S = 40.0              # local-regression time bandwidth, tuned on flight 01
N_ITERS = 15
TUKEY_C = 4.685                 # standard Tukey biweight tuning constant
MAX_FRAME_GAP = 15              # matches evaluate_ground_truth.py's default


def project_to_local_xy(lats, lngs, lat0, lng0):
    mlng = m_per_deg_lng(lat0)
    x = (lngs - lng0) * mlng
    y = (lats - lat0) * M_PER_DEG_LAT
    return x, y


def unproject_from_local_xy(x, y, lat0, lng0):
    mlng = m_per_deg_lng(lat0)
    return lat0 + y / M_PER_DEG_LAT, lng0 + x / mlng


def local_linear_smooth(t, v, weights, bandwidth_s, query_t=None):
    """Vectorized local-linear regression of v(t) (Gaussian kernel in time),
    weighted by `weights`. Returns fitted values at query_t (defaults to t)."""
    if query_t is None:
        query_t = t
    dt = query_t[:, None] - t[None, :]                      # (nq, n)
    kernel = np.exp(-0.5 * (dt / bandwidth_s) ** 2)
    w = kernel * weights[None, :]
    S0 = w.sum(axis=1)
    S1 = (w * dt).sum(axis=1)
    S2 = (w * dt ** 2).sum(axis=1)
    T0 = (w * v[None, :]).sum(axis=1)
    T1 = (w * dt * v[None, :]).sum(axis=1)
    denom = S0 * S2 - S1 ** 2
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return (S2 * T0 - S1 * T1) / denom


def tukey_biweight(residuals, c=TUKEY_C):
    scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    scale = max(scale, 1e-6)
    u = residuals / (c * scale)
    return np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0), scale


def batch_fuse(t, x, y, bandwidth_s=BANDWIDTH_S, n_iters=N_ITERS):
    weights = np.ones_like(t)
    fx = fy = resid = None
    for _ in range(n_iters):
        fx = local_linear_smooth(t, x, weights, bandwidth_s)
        fy = local_linear_smooth(t, y, weights, bandwidth_s)
        resid = np.hypot(x - fx, y - fy)
        weights, _ = tukey_biweight(resid)
    return fx, fy, weights, resid


def load_raw_track(gt_path, telemetry_path, max_gap=MAX_FRAME_GAP):
    gt = load_ground_truth(gt_path)
    tel = load_telemetry(telemetry_path)
    frames, times, raw_lat, raw_lng, gt_lat, gt_lng = [], [], [], [], [], []
    for pt in gt:
        row = nearest_telemetry_row(tel, pt["frame"])
        if abs(row["frame"] - pt["frame"]) > max_gap:
            continue
        frames.append(pt["frame"]); times.append(pt["time_sec"])
        raw_lat.append(row["raw_lat"]); raw_lng.append(row["raw_lng"])
        gt_lat.append(pt["lat"]); gt_lng.append(pt["lng"])
    return (np.array(frames), np.array(times), np.array(raw_lat), np.array(raw_lng),
            np.array(gt_lat), np.array(gt_lng))


def run_flight(flight):
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    tel_path = CSV_DIR / f"video_telemetry_orb_uavvisloc_uavvisloc_{flight}.csv"

    frames, t, raw_lat, raw_lng, gt_lat, gt_lng = load_raw_track(gt_path, tel_path)
    lat0, lng0 = gt_lat.mean(), gt_lng.mean()
    x, y = project_to_local_xy(raw_lat, raw_lng, lat0, lng0)
    gx, gy = project_to_local_xy(gt_lat, gt_lng, lat0, lng0)

    fx, fy, weights, resid_to_fit = batch_fuse(t, x, y)
    fused_lat, fused_lng = unproject_from_local_xy(fx, fy, lat0, lng0)

    raw_err = np.hypot(x - gx, y - gy)
    fused_err = np.hypot(fx - gx, fy - gy)

    existing = compute_stats(str(gt_path), str(tel_path), max_frame_gap=MAX_FRAME_GAP)

    def summarize(errs):
        return error_stats(list(errs))

    raw_stats = summarize(raw_err)
    fused_stats = summarize(fused_err)

    print(f"\n{'=' * 90}\nFlight {flight}  (n={len(t)})\n{'=' * 90}")
    print(f"{'':>22} {'mean':>8} {'median':>8} {'max':>8} {'A@10m':>7} {'A@20m':>7}")

    def row(label, s):
        a = s["a_at"]
        print(f"{label:>22} {s['mean']:7.1f}m {s['median']:7.1f}m {s['max']:7.1f}m "
              f"{a[10]:6.1%} {a[20]:6.1%}")

    row("Raw (this script)", raw_stats)
    if existing:
        row("Raw (evaluate_gt.py)", existing["raw"])
        row("Kalman-filtered", existing["filt"])
    row("Batch-fused (IRLS)", fused_stats)

    # Sanity-check the mechanism: do high-final-weight frames actually have lower
    # TRUE error (ground truth used only for this check, never fed into the fit)?
    order = np.argsort(weights)
    n = len(weights)
    low_third = order[: n // 3]
    high_third = order[-(n // 3):]
    print(f"\nMechanism check -- true raw error by final IRLS weight tercile:")
    print(f"  low-weight third  (n={len(low_third)}):  mean raw error = {raw_err[low_third].mean():.1f}m")
    print(f"  high-weight third (n={len(high_third)}): mean raw error = {raw_err[high_third].mean():.1f}m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt_lng, gt_lat, "-", color="#1f77b4", linewidth=1.5, label="ground truth", zorder=2)
    ax.scatter(raw_lng, raw_lat, c=weights, cmap="autumn_r", s=10, alpha=0.6,
               label="raw ORB guess (color = final IRLS weight)", zorder=3)
    ax.plot(fused_lng, fused_lat, "-", color="#2ca02c", linewidth=2.0,
            label="batch-fused trajectory", zorder=4)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Flight {flight}: batch trajectory fusion vs. ground truth")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"traj_flight{flight}.png", dpi=130)
    plt.close(fig)

    return raw_stats, fused_stats, existing


def main():
    for flight in FLIGHTS:
        run_flight(flight)
    print(f"\nPlots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
