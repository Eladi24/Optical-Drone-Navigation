#!/usr/bin/env python3
"""
Synthetic check for stills_odometry_fusion.local_consensus(offset_model="loclin") (2026-11-24).

Models the mechanism measured on the design set (INVESTIGATION_LOG "Fusion-Tail Diagnostic" + follow-up): on most
drift stills the winning cluster is ONE-SIDED (all members before or after the still), so the flat mean of their
offsets is the offset some distance away, and the odometry drift over that distance becomes the error. (A first
version of this check used symmetric windows with linear drift; there the flat mean is already exact -- the mean of a
line over a symmetric span is its centre value -- so it could not separate the models. That was a fault in the test's
premise, not in the method.)

Scenario: a straight ~8 km track of 100 stills, 80 m apart; odometry with a constant heading bias (linear drift);
anchors = truth + 15 m noise, 20% replaced by far outliers; NO anchors on the last 8 stills (an end-of-flight gap,
like flight 08's), so the clusters there are one-sided. Asserts:
  - with a 4 deg bias, on the gap stills loclin's median error is < half the flat mean's;
  - with no bias, the two agree within 2 m overall (nothing to fix -> no cost).
Exit code 0 = pass.

    python3 -I scripts/check_fusion_loclin.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import stills_odometry_fusion as sof  # noqa: E402

N, STEP, GAP = 100, 80.0, 8


def run(bias_deg, model, seed=7):
    rng = np.random.default_rng(seed)
    truth = np.column_stack([np.arange(N) * STEP, np.zeros(N)])
    th = np.radians(bias_deg)
    odo = np.tile([STEP * np.cos(th), STEP * np.sin(th)], (N - 1, 1))   # biased heading
    p = sof.dead_reckon(odo)
    path_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(odo, axis=1))])
    a_idx = np.arange(N - GAP)                                           # end-of-flight anchor gap
    anchors = truth[a_idx] + rng.normal(0, 15.0, (len(a_idx), 2))
    bad = rng.random(len(a_idx)) < 0.2
    anchors[bad] += rng.uniform(-3000, 3000, (bad.sum(), 2))
    est = sof.local_consensus(p, a_idx, anchors[:, 0], anchors[:, 1], half=10, path_s=path_s, offset_model=model)
    err = np.linalg.norm(est - truth, axis=1)
    gap = np.arange(N - GAP, N)
    gap_ok = gap[~np.isnan(err[gap])]
    return float(np.nanmedian(err)), float(np.median(err[gap_ok])) if gap_ok.size else float("nan"), gap_ok.size


def main():
    failures = []
    a_all, a_gap, n_gap = run(4.0, "mean")
    b_all, b_gap, _ = run(4.0, "loclin")
    print(f"4 deg bias  | all stills median: mean {a_all:6.1f} m, loclin {b_all:6.1f} m | "
          f"one-sided gap stills ({n_gap} covered): mean {a_gap:6.1f} m, loclin {b_gap:6.1f} m")
    if not b_gap < 0.5 * a_gap:
        failures.append("bias case: expected loclin < half the flat mean on the one-sided gap stills")
    z_all, _, _ = run(0.0, "mean")
    zl_all, _, _ = run(0.0, "loclin")
    print(f"no bias     | all stills median: mean {z_all:6.1f} m, loclin {zl_all:6.1f} m")
    if abs(z_all - zl_all) > 2.0:
        failures.append("no-bias case: expected the two models within 2 m")
    for f in failures:
        print("FAIL:", f)
    print("PASS" if not failures else "FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
