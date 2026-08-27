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

STEP 2 (added this pass): STRATEGY.md's second batch-estimator sub-task, "sliding-window
refinement via bounded weighted Procrustes updates". Step 1's per-axis LOESS fit smooths
x(t) and y(t) completely independently, so nothing in it can express or correct a
*rotational* misalignment -- only independent per-axis translational bias. This adds
`windowed_procrustes_refine()`: within overlapping time windows (WINDOW_S, 50% stride) over
Step 1's already-fitted curve, solve a weighted RIGID (rotation + translation, no scale --
both point sets are already in real-world metres, and STRATEGY.md's resolution-gap
experiment already ruled out a scale/resolution effect) alignment via closed-form weighted
Kabsch/SVD between Step 1's fitted points and that window's raw matches, wrapped in a small
IRLS loop (reusing tukey_biweight) so within-window outliers get down-weighted the same way
Step 1 already does. The resulting per-window correction is BOUNDED before being applied:
rotation clipped to ROTATION_CAP_DEG (untuned default, same honesty convention as Step 1's
Tukey constant), translation clipped to a cap computed per-flight from that flight's own
real GT track speed (offline-diagnostic-only calibration -- a fielded system would use a
known platform speed limit instead of per-flight ground truth). Overlapping windows are
blended via a triangular taper. See .claude/plans/iterative-launching-parrot.md for the
full design rationale and CLAUDE.md for the measured result.

STEP 3 (added this pass): STRATEGY.md's third batch-estimator sub-task, "closed-form MAP
estimate fusing odometry displacements with retrieval anchors, clamping detected outliers."
`map_fuse_trajectory()` below is a generic, dataset-agnostic closed-form MAP/smoother solver
-- purely additive, never called from main()/run_flight()/FLIGHTS, does not alter anything
in the UAV-VisLoc Step 1/2 path above. It is validated as a MECHANISM-ONLY check on MUN-FRL
quarry1's real Kimera-VIO output plus synthesized retrieval-anchor-like corrections, not as a
Phase 4 accuracy result -- no dataset on disk has continuous-video odometry and wide-area
retrieval anchors together (quarry1's whole real flight path fits inside a ~150x30m box, too
small for satellite retrieval to mean anything; UAV-VisLoc has no continuous video or dense
telemetry). See scripts/quarry1_fusion_mechanism_check.py and CLAUDE.md's "Investigation Log:
Phase 4 Step 3" for the full split-validation rationale (same honesty convention as Step 1's
own documented "deliberate substitution, not a literal implementation").

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/batch_trajectory_fusion.py
    # Re-run Step 1/2 against a different estimator's raw telemetry (e.g. to check
    # whether a stronger Phase 2 retrieval+matching combo changes the fusion picture --
    # see CLAUDE.md's "Investigation Log: split_learned_disk through batch fusion"):
    python3 scripts/batch_trajectory_fusion.py --estimator split_learned_disk
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

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
DEFAULT_ESTIMATOR = "orb"       # which telemetry CSV family to fuse -- orb is this
                                 # script's original, hyperparameter-tuning baseline;
                                 # override via --estimator to fuse a different Phase 2
                                 # config's raw matches without retuning anything below
BANDWIDTH_S = 40.0              # local-regression time bandwidth, tuned on flight 01
N_ITERS = 15
TUKEY_C = 4.685                 # standard Tukey biweight tuning constant
MAX_FRAME_GAP = 15              # matches evaluate_ground_truth.py's default

# Step 2 (windowed Procrustes refinement) hyperparameters -- tuned once on flight 01
# only, then held fixed across validation flights, same discipline as Step 1's own
# BANDWIDTH_S/TUKEY_C above.
WINDOW_S = 150.0                 # ~3.75x Step 1's bandwidth -- enough points/window to
                                  # fit a 2D rigid transform (rotation+translation) robustly
WINDOW_IRLS_ITERS = 3
MIN_WINDOW_POINTS = 6            # skip a window with too few points to fit reliably
ROTATION_CAP_DEG = 20.0          # untuned default, same honesty convention as TUKEY_C
TRANSLATION_CAP_PERCENTILE = 95.0
TRANSLATION_CAP_SAFETY_FACTOR = 3.0

# Step 3 (closed-form MAP fusion) hyperparameters -- generic solver defaults, not tuned
# against any single flight (unlike Step 1/2's BANDWIDTH_S/ROTATION_CAP_DEG, which were
# tuned once on flight 01); see quarry1_fusion_mechanism_check.py for the mechanism-check
# -specific anchor-synthesis parameters, which ARE grounded/flagged there.
MAP_IRLS_ITERS = 15              # matches Step 1's N_ITERS convention
MAP_RIDGE = 1e-6                 # tiny Tikhonov term for numerical well-posedness only --
                                  # NOT a modeling choice (see map_fuse_trajectory docstring)


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


def weighted_rigid_align(P, Q, weights):
    """Closed-form weighted rigid (rotation + translation, no scale) alignment mapping
    P -> Q (weighted Kabsch/Umeyama via SVD). Returns (theta_rad, tx, ty, R) such that
    Q ~= R @ P + [tx, ty] in the weighted-least-squares sense."""
    w = np.asarray(weights, dtype=float)
    wsum = w.sum()
    w = w / wsum if wsum > 1e-9 else np.full_like(w, 1.0 / len(w))
    p_bar = (w[:, None] * P).sum(axis=0)
    q_bar = (w[:, None] * Q).sum(axis=0)
    Pc, Qc = P - p_bar, Q - q_bar
    H = Pc.T @ (w[:, None] * Qc)
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    d = 1.0 if np.linalg.det(V @ U.T) >= 0 else -1.0
    R = V @ np.diag([1.0, d]) @ U.T
    t = q_bar - R @ p_bar
    theta = np.arctan2(R[1, 0], R[0, 0])
    return theta, t[0], t[1], R


def compute_translation_cap(t, gx, gy, window_s,
                             percentile=TRANSLATION_CAP_PERCENTILE,
                             safety_factor=TRANSLATION_CAP_SAFETY_FACTOR):
    """Per-flight Step-2 translation bound, grounded in that flight's own real GT
    track speed rather than a guessed constant (matches this project's established
    practice, e.g. MIN_VALID_KEYPOINTS/MIN_CROP_CONTRAST) -- offline-diagnostic-only
    calibration; a fielded system would use a known platform speed limit instead."""
    order = np.argsort(t)
    ts, gxs, gys = t[order], gx[order], gy[order]
    dt = np.diff(ts)
    valid = dt > 0
    if not np.any(valid):
        return window_s * 50.0  # fallback, shouldn't happen with real dense GT
    dist = np.hypot(np.diff(gxs), np.diff(gys))[valid]
    speeds = dist / dt[valid]
    cap_speed = safety_factor * np.percentile(speeds, percentile)
    return cap_speed * window_s


def windowed_procrustes_refine(t, x, y, fx, fy, translation_cap_m,
                                window_s=WINDOW_S, rotation_cap_deg=ROTATION_CAP_DEG,
                                n_iters=WINDOW_IRLS_ITERS, min_points=MIN_WINDOW_POINTS):
    """Phase 4 Step 2: sliding-window bounded weighted Procrustes refinement of Step
    1's fitted trajectory (fx, fy) against the raw matches (x, y). See the module
    docstring and .claude/plans/iterative-launching-parrot.md for design rationale.

    Returns (rx, ry, blended_weight) -- refined trajectory plus a per-frame blended
    robustness weight (for the mechanism-check report), falling back to Step 1's own
    (fx, fy) at any frame no window covers."""
    t_min, t_max = t.min(), t.max()
    stride = window_s / 2.0
    centers = np.arange(t_min + stride, t_max - stride + stride, stride)
    if len(centers) == 0:
        centers = np.array([(t_min + t_max) / 2.0])

    rotation_cap = np.deg2rad(rotation_cap_deg)
    accum = np.zeros((len(t), 2))
    accum_w = np.zeros(len(t))
    accum_robust_w = np.zeros(len(t))

    for c in centers:
        idx = np.where(np.abs(t - c) <= window_s / 2.0)[0]
        if len(idx) < min_points:
            continue
        P = np.stack([fx[idx], fy[idx]], axis=1)
        Q = np.stack([x[idx], y[idx]], axis=1)
        w = np.ones(len(idx))
        theta = tx = ty = 0.0
        R = np.eye(2)
        for _ in range(n_iters):
            theta, tx, ty, R = weighted_rigid_align(P, Q, w)
            resid = np.linalg.norm((R @ P.T).T + np.array([tx, ty]) - Q, axis=1)
            new_w, _ = tukey_biweight(resid)
            if new_w.sum() < 1e-9:
                # Reweighting collapsed to zero (degenerate window) -- keep this
                # iteration's transform rather than feeding an all-zero/NaN weight
                # vector into the next solve.
                break
            w = new_w

        theta_b = np.clip(theta, -rotation_cap, rotation_cap)
        t_mag = np.hypot(tx, ty)
        if t_mag > translation_cap_m:
            scale = translation_cap_m / t_mag
            tx, ty = tx * scale, ty * scale
        Rb = np.array([[np.cos(theta_b), -np.sin(theta_b)],
                        [np.sin(theta_b), np.cos(theta_b)]])
        refined = (Rb @ P.T).T + np.array([tx, ty])

        taper = np.clip(1.0 - np.abs(t[idx] - c) / (window_s / 2.0), 0.0, 1.0)
        accum[idx] += refined * taper[:, None]
        accum_w[idx] += taper
        accum_robust_w[idx] += taper * w

    covered = accum_w > 1e-9
    rx, ry, blended_w = fx.copy(), fy.copy(), np.zeros(len(t))
    rx[covered] = accum[covered, 0] / accum_w[covered]
    ry[covered] = accum[covered, 1] / accum_w[covered]
    blended_w[covered] = accum_robust_w[covered] / accum_w[covered]
    return rx, ry, blended_w


def map_fuse_trajectory(t, odo_dx, odo_dy, anchor_idx, anchor_x, anchor_y,
                         sigma_odo=1.0, sigma_anchor=20.0,
                         n_iters=MAP_IRLS_ITERS, ridge=MAP_RIDGE):
    """Phase 4 Step 3: closed-form MAP fusion of consecutive-keyframe odometry
    displacements with sparse absolute anchor observations, IRLS-robustified
    (Tukey biweight, reusing tukey_biweight() -- the same idiom Step 1/2 already
    use) against outliers in either the odometry chain (e.g. a VIO bias-freeze/
    reset segment) or the anchors (e.g. a wrong-crop-style retrieval outlier).

    Per-axis linear MAP solve, equivalent to a discrete-time RTS/Kalman-smoother
    estimate for a random-walk-with-known-control-input process model (the
    control input being the odometry displacement) plus linear-Gaussian absolute
    measurements -- i.e. a 1D pose-graph with relative + absolute factors. With
    weights held fixed, the MAP estimate is the exact minimizer of the weighted
    least-squares objective sum_i ||x[i+1]-x[i]-d_i||^2/sigma_odo^2 +
    sum_k ||x[anchor_idx[k]]-a_k||^2/sigma_anchor^2, obtained via ONE sparse
    linear solve of the normal equations (A^T W A + ridge*I) v = A^T W b per
    axis. Each IRLS iteration re-solves this exactly -- never an iterative
    nonlinear optimizer (there are no rotation unknowns here, unlike Step 2's
    Procrustes) -- then reweights odometry and anchor residuals independently
    via tukey_biweight() on their POOLED joint (x,y) residual norm, so a
    factor that's bad in either axis is flagged.

    `ridge` is a tiny Tikhonov term for numerical well-posedness only (odometry
    differences alone are rank-deficient by one d.o.f. per axis without at
    least one anchor, and IRLS can zero out every anchor in a given iteration)
    -- not a modeling / prior-belief choice.

    Args:
        t: (N,) VIO keyframe times (seconds, any consistent epoch) -- carried
            through only for API symmetry with the rest of this file; the
            solve itself is time-implicit (odometry displacements are already
            per-keyframe increments).
        odo_dx, odo_dy: (N-1,) consecutive-keyframe displacements, ALREADY in
            the target/world frame -- any VIO-frame rotation/scale correction
            must happen upstream of this function, which stays a general,
            dataset-agnostic MAP solver.
        anchor_idx: (M,) int, which state index (0..N-1) each anchor attaches to.
        anchor_x, anchor_y: (M,) anchor absolute positions, target/world frame.
        sigma_odo, sigma_anchor: nominal per-factor noise std (metres) --
            untuned defaults, same honesty convention as ROTATION_CAP_DEG.
        n_iters: IRLS iteration count.
        ridge: Tikhonov regularization (numerical well-posedness only).

    Returns:
        fx, fy: (N,) fused per-keyframe position estimate.
        w_odo: (N-1,) final IRLS weight per odometry factor.
        w_anchor: (M,) final IRLS weight per anchor factor.
    """
    N = len(t)
    n_odo = N - 1
    M = len(anchor_idx)
    anchor_idx = np.asarray(anchor_idx, dtype=int)

    # Fixed sparsity pattern (built once; only the weights change per iteration).
    rows_odo = np.repeat(np.arange(n_odo), 2)
    cols_odo = np.column_stack([np.arange(n_odo), np.arange(1, N)]).ravel()
    vals_odo = np.tile([-1.0, 1.0], n_odo)
    rows_anc = np.arange(n_odo, n_odo + M)
    cols_anc = anchor_idx
    vals_anc = np.ones(M)
    A = sp.csr_matrix(
        (np.concatenate([vals_odo, vals_anc]),
         (np.concatenate([rows_odo, rows_anc]), np.concatenate([cols_odo, cols_anc]))),
        shape=(n_odo + M, N))
    ridge_I = ridge * sp.identity(N, format="csr")
    b_x = np.concatenate([odo_dx, anchor_x])
    b_y = np.concatenate([odo_dy, anchor_y])

    w_odo = np.ones(n_odo)
    w_anchor = np.ones(M)
    fx = fy = None

    for _ in range(n_iters):
        Wd = np.concatenate([w_odo / sigma_odo ** 2, w_anchor / sigma_anchor ** 2])
        Wsp = sp.diags(Wd)
        AtWA = (A.T @ Wsp @ A + ridge_I).tocsc()

        fx = spsolve(AtWA, A.T @ (Wd * b_x))
        fy = spsolve(AtWA, A.T @ (Wd * b_y))

        odo_resid_x = (fx[1:] - fx[:-1]) - odo_dx
        odo_resid_y = (fy[1:] - fy[:-1]) - odo_dy
        anc_resid_x = fx[anchor_idx] - anchor_x
        anc_resid_y = fy[anchor_idx] - anchor_y

        w_odo, _ = tukey_biweight(np.hypot(odo_resid_x, odo_resid_y))
        new_w_anchor, _ = tukey_biweight(np.hypot(anc_resid_x, anc_resid_y))
        if new_w_anchor.sum() > 1e-9:
            # Same degenerate-collapse guard windowed_procrustes_refine() uses --
            # don't feed an all-zero weight vector into the next solve.
            w_anchor = new_w_anchor

    return fx, fy, w_odo, w_anchor


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


def run_flight(flight, estimator=DEFAULT_ESTIMATOR):
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    tel_path = CSV_DIR / f"video_telemetry_{estimator}_uavvisloc_uavvisloc_{flight}.csv"

    frames, t, raw_lat, raw_lng, gt_lat, gt_lng = load_raw_track(gt_path, tel_path)
    lat0, lng0 = gt_lat.mean(), gt_lng.mean()
    x, y = project_to_local_xy(raw_lat, raw_lng, lat0, lng0)
    gx, gy = project_to_local_xy(gt_lat, gt_lng, lat0, lng0)

    fx, fy, weights, resid_to_fit = batch_fuse(t, x, y)
    fused_lat, fused_lng = unproject_from_local_xy(fx, fy, lat0, lng0)

    translation_cap_m = compute_translation_cap(t, gx, gy, WINDOW_S)
    rx, ry, window_weights = windowed_procrustes_refine(t, x, y, fx, fy, translation_cap_m)
    refined_lat, refined_lng = unproject_from_local_xy(rx, ry, lat0, lng0)

    raw_err = np.hypot(x - gx, y - gy)
    fused_err = np.hypot(fx - gx, fy - gy)
    refined_err = np.hypot(rx - gx, ry - gy)

    existing = compute_stats(str(gt_path), str(tel_path), max_frame_gap=MAX_FRAME_GAP)

    def summarize(errs):
        return error_stats(list(errs))

    raw_stats = summarize(raw_err)
    fused_stats = summarize(fused_err)
    refined_stats = summarize(refined_err)

    print(f"\n{'=' * 90}\nFlight {flight}  estimator={estimator}  (n={len(t)})\n{'=' * 90}")
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
    row("Windowed Procrustes", refined_stats)

    # Sanity-check the mechanism: do high-final-weight frames actually have lower
    # TRUE error (ground truth used only for this check, never fed into the fit)?
    order = np.argsort(weights)
    n = len(weights)
    low_third = order[: n // 3]
    high_third = order[-(n // 3):]
    print(f"\nMechanism check (Step 1) -- true raw error by final IRLS weight tercile:")
    print(f"  low-weight third  (n={len(low_third)}):  mean raw error = {raw_err[low_third].mean():.1f}m")
    print(f"  high-weight third (n={len(high_third)}): mean raw error = {raw_err[high_third].mean():.1f}m")

    order2 = np.argsort(window_weights)
    low2, high2 = order2[: n // 3], order2[-(n // 3):]
    print(f"Mechanism check (Step 2, translation cap={translation_cap_m:.0f}m) -- "
          f"true raw error by blended window-weight tercile:")
    print(f"  low-weight third  (n={len(low2)}):  mean raw error = {raw_err[low2].mean():.1f}m")
    print(f"  high-weight third (n={len(high2)}): mean raw error = {raw_err[high2].mean():.1f}m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt_lng, gt_lat, "-", color="#1f77b4", linewidth=1.5, label="ground truth", zorder=2)
    ax.scatter(raw_lng, raw_lat, c=weights, cmap="autumn_r", s=10, alpha=0.6,
               label="raw ORB guess (color = final IRLS weight)", zorder=3)
    ax.plot(fused_lng, fused_lat, "-", color="#2ca02c", linewidth=2.0,
            label="batch-fused trajectory (Step 1)", zorder=4)
    ax.plot(refined_lng, refined_lat, "-", color="#d62728", linewidth=2.0,
            label="windowed Procrustes (Step 2)", zorder=5)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Flight {flight} ({estimator}): batch trajectory fusion vs. ground truth")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    # Keep the default (orb) run's filenames stable -- they're referenced directly
    # elsewhere in CLAUDE.md -- and only suffix non-default estimators so a
    # different config's plots don't clobber them.
    suffix = "" if estimator == DEFAULT_ESTIMATOR else f"_{estimator}"
    fig.savefig(OUT_DIR / f"traj_flight{flight}{suffix}.png", dpi=130)
    plt.close(fig)

    return raw_stats, fused_stats, refined_stats, existing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", default=DEFAULT_ESTIMATOR,
                         help=f"telemetry CSV family to fuse (default: {DEFAULT_ESTIMATOR})")
    parser.add_argument("--flights", default=None,
                         help=f"comma-separated flight numbers to fuse (default: the dev+validation "
                              f"set {FLIGHTS} this script's hyperparameters were tuned/validated on -- "
                              f"pass e.g. 02,05,06,11 to run held-out flights explicitly; this is never "
                              f"the default, matching this project's own discipline of not touching "
                              f"held-out data casually)")
    args = parser.parse_args()
    flights = args.flights.split(",") if args.flights else FLIGHTS

    all_raw, all_fused, all_refined = [], [], []
    for flight in flights:
        raw_stats, fused_stats, refined_stats, _ = run_flight(flight, estimator=args.estimator)
        all_raw.append(raw_stats["mean"]); all_fused.append(fused_stats["mean"]); all_refined.append(refined_stats["mean"])

    print(f"\n{'=' * 90}\nCombined across {flights} (mean-of-per-flight-means, estimator={args.estimator})\n{'=' * 90}")
    print(f"  Raw mean:               {np.mean(all_raw):8.1f}m")
    print(f"  Batch-fused (Step 1) mean: {np.mean(all_fused):8.1f}m")
    print(f"  Windowed Procrustes (Step 2) mean: {np.mean(all_refined):8.1f}m")
    print(f"\nPlots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
