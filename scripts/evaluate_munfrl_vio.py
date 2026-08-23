#!/usr/bin/env python3
"""
STRATEGY.md Phase 3 gate: "Drift as % of distance travelled over 60s windows,
on a dataset with GPS-INS ground truth." Compares Kimera-VIO's estimated
trajectory (Datasets/MUN_FRL_quarry1/kimera_output/traj_vio.csv) against the
real PPK ground truth this project's own ingestion pass produced
(mav0/state_groundtruth_estimate0/data.csv -- real position, placeholder
identity orientation; see CLAUDE.md for why the placeholder is safe here).

METHODOLOGY, and a real bug found and fixed before trusting a first version of
this script: an initial attempt reused evo's own RPE.rpe_base() formula
directly on (ref, est) SE(3) poses, reasoning that the reference's placeholder
identity orientation would cancel out algebraically. It does NOT: because
every reference orientation is the same fixed identity, RPE's relative-pose
error ends up comparing the ground truth's displacement expressed in the TRUE
GLOBAL frame against the VIO estimate's displacement expressed in ITS OWN
LOCAL, time-varying BODY frame (via the estimate's real, evolving orientation)
-- two different reference frames, not a frame-invariant comparison. This
produced a nonsense ~250% "drift" number that looked like VIO divergence but
was actually a bug in this evaluation script. Caught by sanity-checking
direction, not just magnitude, of early-window displacements against ground
truth before trusting the headline number (see CLAUDE.md).

Fixed by doing what should have been done from the start: a single GLOBAL
rigid (rotation+translation) Umeyama alignment of the whole estimated
trajectory onto ground truth (evo's own PoseTrajectory3D.align(), the
purpose-built tool for exactly this), computed once over the whole matched
span -- then computing drift as plain displacement-vector error between the
ALIGNED estimate and ground truth over each 60s window, entirely avoiding the
relative-pose/reference-frame issue above. Reported both WITHOUT scale
correction (rigid only) and WITH Umeyama scale correction, since monocular VIO
has a well-known inherent scale-observability weakness under gentle,
non-aggressive motion (this is a real, expected monocular limitation, not
specific to this dataset or Kimera) -- comparing both tells us whether this
run's error is dominated by a global scale bias (a monocular-VIO-typical
issue) or by direction/shape error (more consistent with a tracking/
calibration problem).

Why this isn't a plain `evo_rpe ... -u s` CLI call in the first place: evo
1.37.0 defines Unit.seconds as an enum member, but its actual delta-pairing
code (metrics.id_pairs_from_delta) only implements frames/meters/degrees/
radians -- seconds raises "unsupported delta unit" if you try it, confirmed by
reading evo's own source before trusting the CLI flag.

Usage:
    source ../CV_IP/cv_env/bin/activate
    python3 scripts/evaluate_munfrl_vio.py
"""
import copy
import csv
from pathlib import Path

import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync

REPO = Path(__file__).parent.parent
VIO_TRAJ = REPO / "Datasets" / "MUN_FRL_quarry1" / "kimera_output" / "traj_vio.csv"
GT_TRAJ = REPO / "Datasets" / "MUN_FRL_quarry1" / "mav0" / "state_groundtruth_estimate0" / "data.csv"

WINDOW_S = 60.0
WINDOW_MATCH_TOL_S = 1.0    # accept a "60s later" pose within +-1s of exact
STRIDE_S = 5.0              # slide the window start every 5s for more samples
SYNC_MAX_DIFF_S = 0.15      # ref/est association tolerance


def load_traj(path, ts_is_ns=True):
    ts, pos, quat_wxyz = [], [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            ts_val = float(row[0])
            ts.append(ts_val / 1e9 if ts_is_ns else ts_val)
            pos.append([float(row[1]), float(row[2]), float(row[3])])
            quat_wxyz.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    return PoseTrajectory3D(
        positions_xyz=np.array(pos),
        orientations_quat_wxyz=np.array(quat_wxyz),
        timestamps=np.array(ts),
    )


def windowed_drift(traj_ref, traj_est_aligned):
    """Drift as % of GT distance travelled, per 60s window, using plain
    displacement-vector error (both trajectories now in the same global
    frame post-alignment) -- see module docstring for why this replaced the
    original relative-pose (RPE) approach."""
    ts = traj_ref.timestamps
    ref_pos = traj_ref.positions_xyz
    est_pos = traj_est_aligned.positions_xyz
    ref_dist_cum = traj_ref.distances

    window_starts = np.arange(ts[0], ts[-1] - WINDOW_S, STRIDE_S)
    trans_errors, ratios, window_dists = [], [], []
    for t_start in window_starts:
        i = int(np.argmin(np.abs(ts - t_start)))
        target_t = ts[i] + WINDOW_S
        j = int(np.argmin(np.abs(ts - target_t)))
        if abs(ts[j] - target_t) > WINDOW_MATCH_TOL_S or j <= i:
            continue
        gt_disp = ref_pos[j] - ref_pos[i]
        est_disp = est_pos[j] - est_pos[i]
        trans_error = float(np.linalg.norm(est_disp - gt_disp))
        gt_dist = float(ref_dist_cum[j] - ref_dist_cum[i])
        if gt_dist <= 1e-6:
            continue
        trans_errors.append(trans_error)
        window_dists.append(gt_dist)
        ratios.append(trans_error / gt_dist * 100.0)
    return np.array(ratios), np.array(trans_errors), np.array(window_dists)


def report(label, ratios, trans_errors, window_dists):
    print(f"\n--- {label} ---")
    if len(ratios) == 0:
        print("No valid 60s windows.")
        return
    print(f"{len(ratios)} valid windows")
    print(f"Drift %:        mean={ratios.mean():.2f}%  median={np.median(ratios):.2f}%  "
          f"max={ratios.max():.2f}%  min={ratios.min():.2f}%")
    print(f"Position error: mean={trans_errors.mean():.2f}m  "
          f"median={np.median(trans_errors):.2f}m  max={trans_errors.max():.2f}m")
    print(f"GT distance travelled per window: mean={window_dists.mean():.2f}m  "
          f"median={np.median(window_dists):.2f}m")


def main():
    traj_est = load_traj(VIO_TRAJ)
    traj_ref = load_traj(GT_TRAJ)
    print(f"VIO estimate: {traj_est.num_poses} poses, "
          f"{traj_est.timestamps[-1] - traj_est.timestamps[0]:.1f}s span")
    print(f"Ground truth: {traj_ref.num_poses} poses, "
          f"{traj_ref.timestamps[-1] - traj_ref.timestamps[0]:.1f}s span")

    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=SYNC_MAX_DIFF_S)
    print(f"Synchronized: {traj_ref_sync.num_poses} matched pose pairs "
          f"(max_diff={SYNC_MAX_DIFF_S}s)")
    print(f"(a short flight, so the {WINDOW_S:.0f}s-window sample below is thin, "
          f"not statistically rich -- flagged honestly, not glossed over)")

    est_rigid = copy.deepcopy(traj_est_sync)
    r_rigid = est_rigid.align(traj_ref_sync, correct_scale=False)
    report("Rigid alignment (no scale correction)",
           *windowed_drift(traj_ref_sync, est_rigid))

    est_scaled = copy.deepcopy(traj_est_sync)
    r_scaled = est_scaled.align(traj_ref_sync, correct_scale=True)
    scale = r_scaled[2] if isinstance(r_scaled, (tuple, list)) else None
    print(f"\n(Umeyama scale factor applied: {scale})" if scale is not None else "")
    report("With Umeyama scale correction (monocular VIO's scale is "
           "inherently weakly observable under gentle motion -- this isolates "
           "whether error is dominated by a global scale bias)",
           *windowed_drift(traj_ref_sync, est_scaled))

    print(f"\nTotal GT path length (matched span): {traj_ref_sync.path_length:.1f}m")


if __name__ == "__main__":
    main()
