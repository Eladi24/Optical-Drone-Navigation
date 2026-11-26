#!/usr/bin/env python3
"""
Synthetic decoy-attractor injection on DESIGN flights (2026-11-26, INVESTIGATION_LOG "Wrong-Consensus Guard"): design
material for the wrong-consensus mode, which is nearly absent from dev/val (genuinely outvoted clusters only on 08).
Injection protocol and decision rule are declared in the log before the first run. Compares the current default fusion
with and without --decoy-guard on clean and injected anchors. Held-out flights are refused.

    python3 -I scripts/decoy_injection_diag.py
"""
import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import stills_odometry_fusion as sof  # noqa: E402
from fusion_tail_diag import FLIGHTS, HELD_OUT  # noqa: E402

EPISODE_M, COVER, P_DECOY, DECOY_SIGMA, DECOY_INL = 800.0, 0.2, 0.7, 10.0, 40
SEEDS = range(5)
FUSION = dict(min_inliers=16, gate_r=100.0, prior="steplen", fill="bridge", plot=False)


def inject(S, rng):
    S = copy.deepcopy(S)
    G = np.column_stack([S["gx"], S["gy"]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))])
    n_ep = max(1, int(round(COVER * s[-1] / EPISODE_M)))
    starts, tries = [], 0
    while len(starts) < n_ep and tries < 1000:
        tries += 1
        st = rng.uniform(0, max(1.0, s[-1] - EPISODE_M))
        if all(abs(st - o) >= EPISODE_M for o in starts):
            starts.append(st)
    anchors = {int(k): (x, y, q) for k, x, y, q in zip(S["a_idx"], S["ax"], S["ay"], S["a_inl"])}
    ep_mask = np.zeros(len(G), bool)
    for st in starts:
        idx = np.where((s >= st) & (s < st + EPISODE_M))[0]
        if len(idx) == 0:
            continue
        ep_mask[idx] = True
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(400, 1500)
        D = G[idx[len(idx) // 2]] + dist * np.array([np.cos(ang), np.sin(ang)])
        for k in idx:
            if rng.random() < P_DECOY:
                x, y = D + rng.normal(0, DECOY_SIGMA, 2)
                anchors[int(k)] = (x, y, DECOY_INL)
    ks = sorted(anchors)
    S["a_idx"] = np.array(ks, int)
    S["ax"] = np.array([anchors[k][0] for k in ks])
    S["ay"] = np.array([anchors[k][1] for k in ks])
    S["a_inl"] = np.array([anchors[k][2] for k in ks], int)
    return S, ep_mask


def fused_err(fl, est, kw, S, guard, calib):
    out = Path("/tmp") if False else None  # noqa: F841 (no files written)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        per = {}
        tmp = sof.CSV_DIR.parent / ".decoy_tmp.csv"
        sof.run_flight(fl, est, calib, S=S, decoy_guard=guard, per_still_out=str(tmp), **kw, **FUSION)
        import csv
        e = np.full(len(S["gx"]), np.nan)
        for r in csv.DictReader(open(tmp, newline="")):
            if r["Fused_Err_M"]:
                per[int(r["Frame"])] = float(r["Fused_Err_M"])
        tmp.unlink()
    vid = [int(fn) - 1 for fn in S["vid_num"]]
    for i, v in enumerate(vid):
        e[i] = per.get(v, np.nan)
    return e


def stats(e, mask=None):
    v = e[~np.isnan(e)] if mask is None else e[mask & ~np.isnan(e)]
    return (float(v.mean()), float(np.median(v)), float(np.percentile(v, 90))) if len(v) else (np.nan,) * 3


def main():
    import json
    calib = json.load(open(sof.CALIB_PATH))
    assert not (set(FLIGHTS) & HELD_OUT)
    rows = []
    for fl, (est, kw) in FLIGHTS.items():
        S0 = sof.load_flight(fl, est)
        c_off, c_on = stats(fused_err(fl, est, kw, S0, False, calib)), stats(fused_err(fl, est, kw, S0, True, calib))
        inj = {False: [], True: []}
        epi = {False: [], True: []}
        for seed in SEEDS:
            S1, m = inject(S0, np.random.default_rng(1000 + seed))
            for g in (False, True):
                e = fused_err(fl, est, kw, S1, g, calib)
                inj[g].append(stats(e))
                epi[g].append(stats(e, m)[1])
        i_off, i_on = np.mean(inj[False], axis=0), np.mean(inj[True], axis=0)
        e_off, e_on = float(np.mean(epi[False])), float(np.mean(epi[True]))
        rows.append((fl, c_off, c_on, i_off, i_on, e_off, e_on))
        print(f"{fl:4s} CLEAN off {c_off[0]:7.1f}/{c_off[1]:6.1f}/{c_off[2]:7.1f} on {c_on[0]:7.1f}/{c_on[1]:6.1f}/{c_on[2]:7.1f} | "
              f"INJECTED off {i_off[0]:7.1f}/{i_off[1]:6.1f}/{i_off[2]:7.1f} on {i_on[0]:7.1f}/{i_on[1]:6.1f}/{i_on[2]:7.1f} | "
              f"episode median off {e_off:7.1f} on {e_on:7.1f}", flush=True)
    ok1 = all(r[2][0] <= r[1][0] + 1.0 and r[2][1] <= r[1][1] + 1.0 for r in rows)
    comb_off, comb_on = np.mean([r[3][0] for r in rows]), np.mean([r[4][0] for r in rows])
    ep_better = sum(r[6] < r[5] for r in rows)
    ok2 = comb_on < comb_off and ep_better > len(rows) / 2
    print(f"\n(1) clean no-harm (mean & median within +1 m on every flight): {'OK' if ok1 else 'FAIL'}")
    print(f"(2) injected combined mean {comb_off:.1f} -> {comb_on:.1f}; episode median better on {ep_better}/{len(rows)}: "
          f"{'OK' if ok2 else 'FAIL'}")
    print("DECISION:", "proceed to the held-out look" if ok1 and ok2 else "do NOT proceed")


if __name__ == "__main__":
    main()
