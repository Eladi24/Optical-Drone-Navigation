#!/usr/bin/env python3
"""
Fusion-tail diagnostic (2026-11-24): where does the fused error's TAIL come from? Design set only (dev 01/03, val
04/08/10, AerialVL av1/av2); held-out 02/05/06/11 are refused. No new C++ runs: existing telemetry CSVs (each flight's
current-best anchors) through the frozen default fusion of scripts/stills_odometry_fusion.py
(--min-inliers 16 --gate-r 100 --prior steplen --fill bridge; --half 10 on UAV-VisLoc, --half-m 870 on AerialVL).

For every still the final estimate is tagged by its SOURCE:
  anchor     its own anchor, kept because it lies within gate_r of the local consensus
  consensus  the consensus mean (own anchor missing or disagreeing)
  bridge     no consensus: dead-reckoned from covered stills on both sides, blended
  one-sided  no consensus: dead-reckoned from one side (flight start/end)
and errors are decomposed by source: count, mean/median, share of the flight's total error (= contribution to the mean),
and stills > 100 m. Stills whose CONSENSUS is > 100 m off are split by re-running the vote:
  drift     the winning cluster is mostly correct anchors (<= 150 m from GT): odometry bent the offsets
  outvoted  a mostly-wrong cluster won although >= 3 (MIN_SUPPORT) correct anchors were in the window
  starved   a mostly-wrong cluster won and < 3 correct anchors were in the window
Bridge/one-sided stills report the gap length and the VO-failure fraction inside the gap.

Fair LOESS comparison: stills_odometry_fusion.py scores LOESS only at anchor stills but consensus at ALL stills (not
comparable). Here LOESS is interpolated in time to every still and both are scored on the same stills.

    python3 -I scripts/fusion_tail_diag.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import stills_odometry_fusion as sof  # noqa: E402
from batch_trajectory_fusion import batch_fuse  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "Images" / "fusion_tail_diag"
HELD_OUT = {"02", "05", "06", "11"}
# flight -> (telemetry estimator = current-best anchors on disk, window kwargs)
FLIGHTS = {
    "01": ("split_vlad_xfeat_rc", {"half": 10}),
    "03": ("split_vlad_xfeat", {"half": 10}),
    "04": ("split_vlad_xfeat", {"half": 10}),
    "08": ("split_vlad_xfeat", {"half": 10}),
    "10": ("split_vlad_xfeat_fp", {"half": 10}),
    "av1": ("split_vlad_xfeat_fp_rc", {"half_m": 870.0}),
    "av2": ("split_vlad_xfeat_fp_rc", {"half_m": 870.0}),
}
MIN_INL, GATE_R, PRIOR = 16, 100.0, "steplen"
TAIL_M = 100.0
OFFSET_MODEL = "mean"   # --offset-model loclin re-runs the decomposition under the local-linear consensus
SOURCES = ("anchor", "consensus", "bridge", "one-sided")


def analyse(flight, estimator, win, calib):
    S = sof.load_flight(flight, estimator)
    G = np.column_stack([S["gx"], S["gy"]])
    n = len(G)
    odo = sof.predict_steps(S, calib, prior=PRIOR)
    keep = S["a_inl"] >= MIN_INL
    a_idx, ax, ay = S["a_idx"][keep], S["ax"][keep], S["ay"][keep]
    p = sof.dead_reckon(odo)
    path_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(odo, axis=1))])
    half = win.get("half", sof.HALF_WIN)
    half_m = win.get("half_m")
    cons = sof.local_consensus(p, a_idx, ax, ay, half=half, path_s=path_s, half_m=half_m, offset_model=OFFSET_MODEL)
    fused = sof.gate_and_fill(cons, p, a_idx, ax, ay, n, GATE_R, fill="bridge")
    err = np.linalg.norm(fused - G, axis=1)

    # ---- source of each still's estimate (mirrors gate_and_fill's logic)
    A = np.full((n, 2), np.nan)
    A[a_idx] = np.column_stack([ax, ay])
    agree = np.linalg.norm(A - cons, axis=1) <= GATE_R
    has_cons = ~np.isnan(cons[:, 0])
    covered = np.where(has_cons)[0]
    src = np.empty(n, object)
    gap_len = np.zeros(n, int)
    gap_vofail = np.full(n, np.nan)
    for i in range(n):
        if agree[i]:
            src[i] = "anchor"
        elif has_cons[i]:
            src[i] = "consensus"
        else:
            left, right = covered[covered < i], covered[covered > i]
            src[i] = "bridge" if (left.size and right.size) else "one-sided"
            lo = left[-1] if left.size else 0
            hi = right[0] if right.size else n - 1
            gap_len[i] = hi - lo
            gap_vofail[i] = 1.0 - S["vo_ok"][lo:hi].mean() if hi > lo else np.nan

    # ---- wrong consensus: outvoted vs starved (only where a consensus exists)
    a_err = np.hypot(ax - G[a_idx, 0], ay - G[a_idx, 1])
    cons_err = np.linalg.norm(cons - G, axis=1)
    wrong = has_cons & (cons_err > TAIL_M)
    # Re-run the vote for each wrong still to see WHO won (same arithmetic as sof.local_consensus):
    #   drift     the winning cluster is mostly CORRECT anchors, yet the consensus is > 100 m off -> the odometry
    #             across the window bent the offsets (not an anchor-selection failure)
    #   outvoted  the winning cluster is mostly wrong although >= MIN_SUPPORT correct anchors were in the window
    #   starved   the winning cluster is mostly wrong and < MIN_SUPPORT correct anchors were in the window
    o = np.column_stack([ax - p[a_idx, 0], ay - p[a_idx, 1]])
    kind = np.full(n, "", object)
    noise_term = np.full(n, np.nan)
    odo_term = np.full(n, np.nan)
    for i in np.where(wrong)[0]:
        c = (np.where(np.abs(path_s[a_idx] - path_s[i]) <= half_m)[0] if half_m is not None
             else np.where(np.abs(a_idx - i) <= half)[0])
        d = np.linalg.norm(o[c][:, None] - o[c][None], axis=2)
        b = int(np.argmax((d <= sof.R_AGREE).sum(axis=1)))
        members = c[d[b] <= sof.R_AGREE]
        if np.mean(a_err[members] <= sof.R_AGREE) > 0.5:
            kind[i] = "drift"
            # exact split of the consensus error vector: (p_i + mean o_m) - G_i =
            #   [mean o_m - mean(G_m - p_m)]  anchor noise of the cluster members
            # + [mean(G_m - p_m) - (G_i - p_i)]  odometry bend between the members' stills and still i
            gm = G[a_idx[members]] - p[a_idx[members]]
            noise_term[i] = np.linalg.norm(o[members].mean(axis=0) - gm.mean(axis=0))
            odo_term[i] = np.linalg.norm(gm.mean(axis=0) - (G[i] - p[i]))
        elif (a_err[c] <= sof.R_AGREE).sum() >= sof.MIN_SUPPORT:
            kind[i] = "outvoted"
        else:
            kind[i] = "starved"

    # ---- LOESS interpolated in time to every still (fair comparison)
    t_a = S["t"][a_idx].astype(float)
    fx, fy, _, _ = batch_fuse(t_a, ax, ay)
    order = np.argsort(t_a)
    lx = np.interp(S["t"], t_a[order], fx[order])
    ly = np.interp(S["t"], t_a[order], fy[order])
    loess_err = np.hypot(lx - G[:, 0], ly - G[:, 1])

    total = err.sum()
    out = {"flight": flight, "estimator": estimator, "n": n, "vo_ok": float(S["vo_ok"].mean()),
           "fused": sof.stats(err), "loess_all_stills": sof.stats(loess_err),
           "fused_at_anchor_stills": sof.stats(err[a_idx]), "loess_at_anchor_stills": sof.stats(loess_err[a_idx]),
           "by_source": {}, "wrong_consensus": {}}
    for s in SOURCES:
        m = src == s
        e = err[m]
        d = {"count": int(m.sum()), "frac": float(m.mean()),
             "mean": float(e.mean()) if e.size else float("nan"),
             "median": float(np.median(e)) if e.size else float("nan"),
             "share_of_total_error": float(e.sum() / total) if total > 0 else 0.0,
             "tail_count": int((e > TAIL_M).sum())}
        if s in ("bridge", "one-sided") and m.any():
            d["median_gap_stills"] = float(np.median(gap_len[m]))
            d["median_gap_vo_fail"] = float(np.nanmedian(gap_vofail[m]))
            d["tail_median_gap_vo_fail"] = (float(np.nanmedian(gap_vofail[m & (err > TAIL_M)]))
                                            if (m & (err > TAIL_M)).any() else float("nan"))
        out["by_source"][s] = d
    tail = err > TAIL_M
    out["tail_count"] = int(tail.sum())
    out["tail_share_of_total_error"] = float(err[tail].sum() / total) if total > 0 else 0.0
    own = np.full(n, np.nan)
    own[a_idx] = a_err
    dr = kind == "drift"
    out["drift_detail"] = {
        "median_odometry_term_m": float(np.nanmedian(odo_term[dr])) if dr.any() else float("nan"),
        "median_anchor_noise_term_m": float(np.nanmedian(noise_term[dr])) if dr.any() else float("nan"),
        "own_anchor_within_50m": int((dr & (own <= 50)).sum()),     # a correct own anchor the gate threw away
        "own_anchor_missing_or_wrong": int((dr & ~(own <= 50)).sum()),
    }
    out["wrong_consensus"] = {"stills": int(wrong.sum()),
                              **{k: int((kind == k).sum()) for k in ("drift", "outvoted", "starved")},
                              "of_which_final_in_tail": int((wrong & tail).sum()),
                              "final_error_share": {k: float(err[(kind == k)].sum() / total) if total > 0 else 0.0
                                                    for k in ("drift", "outvoted", "starved")}}
    return out, {"err": err, "src": src, "loess_err": loess_err, "t": S["t"]}


def main():
    global OFFSET_MODEL, OUT
    if "--offset-model" in sys.argv:
        OFFSET_MODEL = sys.argv[sys.argv.index("--offset-model") + 1]
        assert OFFSET_MODEL in ("mean", "loclin"), OFFSET_MODEL
        if OFFSET_MODEL != "mean":
            OUT = OUT.parent / f"{OUT.name}_{OFFSET_MODEL}"
    assert not (set(FLIGHTS) & HELD_OUT), "held-out flights are not part of the design set"
    OUT.mkdir(parents=True, exist_ok=True)
    calib = json.load(open(sof.CALIB_PATH))
    results = []
    for fl, (est, win) in FLIGHTS.items():
        r, per = analyse(fl, est, win, calib)
        results.append(r)
        np.savez(OUT / f"per_still_{fl}.npz", err=per["err"], src=per["src"].astype(str),
                 loess_err=per["loess_err"], t=per["t"])
        f, la = r["fused"], r["loess_all_stills"]
        print(f"\n=== {fl} ({est}, n={r['n']}, VO ok {r['vo_ok']:.0%})")
        print(f"  fused (default)      mean {f['mean']:7.1f} med {f['median']:6.1f} p90 {f['p90']:7.1f}  "
              f"A@50 {f['A@50']:5.1%}  stills > {TAIL_M:.0f} m: {r['tail_count']} "
              f"= {r['tail_share_of_total_error']:.0%} of total error")
        print(f"  LOESS, all stills    mean {la['mean']:7.1f} med {la['median']:6.1f} p90 {la['p90']:7.1f}  "
              f"A@50 {la['A@50']:5.1%}")
        fa, lq = r["fused_at_anchor_stills"], r["loess_at_anchor_stills"]
        print(f"  (anchor stills only: fused mean/med {fa['mean']:.1f}/{fa['median']:.1f}, "
              f"LOESS {lq['mean']:.1f}/{lq['median']:.1f})")
        for s in SOURCES:
            d = r["by_source"][s]
            if not d["count"]:
                continue
            extra = ""
            if "median_gap_stills" in d:
                extra = (f"  gap med {d['median_gap_stills']:.0f} stills, VO-fail in gap {d['median_gap_vo_fail']:.0%}"
                         f" (tail stills: {d['tail_median_gap_vo_fail']:.0%})")
            print(f"    {s:10s} {d['count']:4d} ({d['frac']:4.0%})  mean {d['mean']:7.1f} med {d['median']:6.1f}  "
                  f"share of error {d['share_of_total_error']:4.0%}  >100 m: {d['tail_count']:3d}{extra}")
        w = r["wrong_consensus"]
        fs = w["final_error_share"]
        print(f"    wrong consensus (> {TAIL_M:.0f} m): {w['stills']} stills -> drift {w['drift']} "
              f"({fs['drift']:.0%} of error), outvoted {w['outvoted']} ({fs['outvoted']:.0%}), "
              f"starved {w['starved']} ({fs['starved']:.0%})")
        dd = r["drift_detail"]
        if w["drift"]:
            print(f"      drift: median odometry term {dd['median_odometry_term_m']:.0f} m, anchor-noise term "
                  f"{dd['median_anchor_noise_term_m']:.0f} m; own anchor correct (<= 50 m) but vetoed by the gate: "
                  f"{dd['own_anchor_within_50m']} / {w['drift']}")

    print("\n=== design set summary: share of TOTAL fused error by source (mean over flights)")
    for s in SOURCES:
        print(f"  {s:10s} {np.mean([r['by_source'][s]['share_of_total_error'] for r in results]):5.1%}   "
              f"stills {np.mean([r['by_source'][s]['frac'] for r in results]):5.1%}")
    json.dump(results, open(OUT / "summary.json", "w"), indent=2)
    print(f"\nJSON + per-still arrays: {OUT}")


if __name__ == "__main__":
    main()
