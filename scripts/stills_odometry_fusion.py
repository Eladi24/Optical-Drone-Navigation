#!/usr/bin/env python3
"""
RESULTS.md route #1 bridge experiment: does GOOD relative motion + our REAL retrieval
anchors reach tens of metres?

Inputs per flight:
  - CSV Files/stills_vo_uavvisloc_<f>.csv  (scripts/stills_visual_odometry.py: image-frame
    nadir-point displacement between consecutive stills, camera-only; row 2 = full-res size)
  - CSV Files/agl_uavvisloc_<f>.csv  (optional, scripts/compute_uavvisloc_agl.py: per-still
    height above ground -- UAV-VisLoc's `height` is NOT AGL over elevated terrain; without
    this file the raw height is used, fine only for near-sea-level flights 01/02/03/04/08)
  - Datasets/UAV_VisLoc_dataset/<f>/<f>.csv  (full-precision GT lat/lon -- used ONLY for
    evaluation and the one-off dev-flight calibration fit -- plus the GNSS-free sensors a
    real platform has: altitude, IMU attitude Omega/Kappa, compass heading Phi1, time)
  - CSV Files/video_telemetry_<est>_uavvisloc_uavvisloc_<f>.csv  (per-frame retrieval
    anchors; its Frame column is the 0-based video index = image filename number - 1; NOT num - 1,
    which differs on flight 08)

Sensors assumed (standard GNSS-denied suite): camera, altimeter (AGL), IMU attitude, compass.

Method:
  1. Odometry model, fitted on dev flights 01/03 only (Huber LSQ):
       body_step = G @ (agl * img_disp_px / width_px) + B @ d(agl * tan(attitude))
     G absorbs focal length + camera-mount rotation/mirroring (normalising by image width
     lets it transfer to the 3000px-wide flights 05/10); B the tilt-induced nadir shift.
     body (forward, right) -> world (E, N) via compass heading. Failed VO pairs (open water,
     no texture) -> weak prior: compass heading + recent median speed.
  2. Local consensus: dead-reckon the odometry from an arbitrary origin (no GT); each anchor
     votes for an offset o_k = anchor_k - p_k. For each still, among anchors within
     +-HALF_WIN stills, the offset with the most votes within R wins (>= MIN_SUPPORT votes);
     position = p_i + mean of the agreeing offsets. Correct anchors agree on one offset while
     wrong ones scatter -- including repeated "attractor" crops, since the platform moves
     ~80 m between stills while an attractor stays put.

Why local consensus and not a global robust MAP: measured on dev (2026-11-16), a single
pose graph with Tukey-robust anchors collapses when <=10% of anchors are correct (fixed
150 m kernel loses all anchors once the fit is >150 m off), and the odometry carries a
per-flight, heading-dependent compass deviation (4-10 deg between opposite survey legs) that
a global chain cannot absorb without per-leg anchors. Local rigidity is what the odometry
actually provides: 25-50 m RMS over 40 stills (~3 km) after translation+rotation alignment.

Hyperparameters (HALF_WIN, MIN_SUPPORT, R) were swept on dev 01/03 AND validation 04/08/10
-- validation was looked at, so it is not a clean check; held-out 02/05/06/11 is the gate
and is only evaluated with an explicit --flights.

Usage:
  python3 scripts/stills_odometry_fusion.py [--recalibrate] [--flights 01,03,04,08,10]
  python3 scripts/stills_odometry_fusion.py --flights 01,03 --oracle   # anchor-selection floor
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Datasets" / "UAV_VisLoc_dataset"
CSV_DIR = REPO / "CSV Files"
OUT_DIR = REPO / "Images" / "stills_fusion"
CALIB_PATH = CSV_DIR / "stills_vo_calibration.json"
sys.path.insert(0, str(Path(__file__).parent))

DEV = ["01", "03"]
VAL = ["04", "08", "10"]
HELD_OUT = ["02", "05", "06", "11"]
M_LAT = 111320.0

MIN_VO_INLIERS = 30
HALF_WIN = 40          # stills either side
MIN_SUPPORT = 3        # agreeing anchors needed
R_AGREE = 150.0        # half a reference crop -- the same radius recall@k calls "correct"


def m_lng(lat):
    return M_LAT * math.cos(math.radians(lat))


def load_flight(flight, estimator):
    rows = list(csv.DictReader(open(DATA / flight / f"{flight}.csv", newline="")))
    lat0, lng0 = float(rows[0]["lat"]), float(rows[0]["lon"])
    ml = m_lng(lat0)
    n = len(rows)
    S = {
        "num": np.array([int(r["num"]) for r in rows]),
        "gx": np.array([(float(r["lon"]) - lng0) * ml for r in rows]),
        "gy": np.array([(float(r["lat"]) - lat0) * M_LAT for r in rows]),
        "om": np.radians([float(r["Omega"]) for r in rows]),
        "ka": np.radians([float(r["Kappa"]) for r in rows]),
        "hdg": np.radians([float(r["Phi1"]) for r in rows]),
    }
    t = [datetime.fromisoformat(r["date"]) for r in rows]
    S["t"] = np.array([(x - t[0]).total_seconds() for x in t])
    num_to_idx = {v: i for i, v in enumerate(S["num"])}
    # Video Frame -> row: the assembled video follows the FILENAME number (Frame = filename
    # number - 1), not `num` -- they differ on flight 08 (num jumps 220 -> 261).
    vid_to_idx = {int(r["filename"].rsplit("_", 1)[1].split(".")[0]) - 1: i for i, r in enumerate(rows)}
    S["vid_num"] = [r["filename"].rsplit("_", 1)[1].split(".")[0] for r in rows]

    agl_path = CSV_DIR / f"agl_uavvisloc_{flight}.csv"
    if agl_path.exists():
        agl = {int(r["Num"]): float(r["AGL"]) for r in csv.DictReader(open(agl_path, newline=""))}
        S["alt"] = np.array([agl[k] for k in S["num"]])
        S["alt_source"] = "AGL (height - DEM)"
    else:
        S["alt"] = np.array([float(r["height"]) for r in rows])
        S["alt_source"] = "raw height"

    vo_ok = np.zeros(n - 1, bool)
    vo_d = np.zeros((n - 1, 2))
    with open(CSV_DIR / f"stills_vo_uavvisloc_{flight}.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["From_Num"].startswith("#"):
                S["width_px"] = float(r["To_Num"])
                continue
            i = num_to_idx[int(r["From_Num"])]
            if num_to_idx[int(r["To_Num"])] != i + 1:
                continue
            if r["Disp_X_Px"] and int(r["Inliers"]) >= MIN_VO_INLIERS:
                vo_ok[i] = True
                vo_d[i] = float(r["Disp_X_Px"]), float(r["Disp_Y_Px"])
    S["vo_ok"], S["vo_d"] = vo_ok, vo_d

    a_idx, ax, ay, ainl = [], [], [], []
    tel = CSV_DIR / f"video_telemetry_{estimator}_uavvisloc_uavvisloc_{flight}.csv"
    with open(tel, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Measurement_Valid", "1") not in ("1", "true", "True"):
                continue
            vid = int(r["Frame"])
            if vid not in vid_to_idx:
                continue
            a_idx.append(vid_to_idx[vid])
            ax.append((float(r["Raw_Lng"]) - lng0) * ml)
            ay.append((float(r["Raw_Lat"]) - lat0) * M_LAT)
            ainl.append(int(r["Inliers"]) if r.get("Inliers") else 0)
    S["a_idx"], S["ax"], S["ay"] = np.array(a_idx, int), np.array(ax), np.array(ay)
    S["a_inl"] = np.array(ainl, int)
    return S


# ---------------------------------------------------------------- odometry model
def design(S):
    a, w = S["alt"], S["width_px"]
    return np.column_stack([
        a[:-1] * S["vo_d"][:, 0] / w, a[:-1] * S["vo_d"][:, 1] / w,
        np.diff(a * np.tan(S["om"])), np.diff(a * np.tan(S["ka"])),
    ])


def gt_body_steps(S):
    dE, dN = np.diff(S["gx"]), np.diff(S["gy"])
    h = S["hdg"][:-1]
    return np.column_stack([dN * np.cos(h) + dE * np.sin(h), dE * np.cos(h) - dN * np.sin(h)])


def body_to_world(body, hdg):
    F, R = body[:, 0], body[:, 1]
    return np.column_stack([F * np.sin(hdg) + R * np.cos(hdg), F * np.cos(hdg) - R * np.sin(hdg)])


def fit_calibration(flights, estimator):
    Xs, Ys = [], []
    for fl in flights:
        S = load_flight(fl, estimator)
        Xs.append(design(S)[S["vo_ok"]]); Ys.append(gt_body_steps(S)[S["vo_ok"]])
    X, Y = np.vstack(Xs), np.vstack(Ys)
    w = np.ones(len(X))
    for _ in range(10):                       # Huber IRLS on the joint residual norm
        sw = np.sqrt(w)[:, None]
        W, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)
        r = np.linalg.norm(Y - X @ W, axis=1)
        k = 1.345 * 1.4826 * np.median(r)
        w = np.where(r <= k, 1.0, k / np.maximum(r, 1e-9))
    r = np.linalg.norm(Y - X @ W, axis=1)
    G = W[:2]
    hfov = 2 * math.degrees(math.atan(0.5 * math.sqrt(abs(np.linalg.det(G)))))
    return {"W": W.tolist(), "flights": flights, "n_pairs": int(len(X)),
            "fit_resid_median_m": float(np.median(r)), "fit_resid_p90_m": float(np.percentile(r, 90)),
            "implied_hfov_deg": hfov}


def predict_steps(S, calib, prior="speed"):
    """Failed VO pairs get a weak prior along the compass heading. prior="speed" (the original):
    recent median speed * dt. prior="steplen" (2026-11-22): recent median step LENGTH, ignoring dt.
    The speed prior explodes on capture pauses: flight 08 step 213 has dt = 568 s and became a
    14.8 km phantom step (GT: 299 m), which was the whole 08 consensus blow-up (p90 14.5 km). The
    surveys capture at ~constant spacing whatever dt is, so step length is the better prior."""
    body = design(S) @ np.array(calib["W"])
    dt = np.diff(S["t"])
    ok = S["vo_ok"]
    step_len = np.linalg.norm(body, axis=1)
    rate = step_len if prior == "steplen" else step_len / np.maximum(dt, 1e-3)
    fallback = float(np.median(rate[ok])) if ok.any() else (80.0 if prior == "steplen" else 20.0)
    recent, last = np.empty(len(body)), []
    for i in range(len(body)):
        if ok[i]:
            last = (last + [rate[i]])[-5:]
        recent[i] = np.median(last) if last else fallback
    prior_len = recent if prior == "steplen" else recent * dt
    body = np.where(ok[:, None], body, np.column_stack([prior_len, np.zeros(len(dt))]))
    return body_to_world(body, S["hdg"][:-1])


def dead_reckon(odo, start=(0.0, 0.0)):
    return np.vstack([start, np.asarray(start) + np.cumsum(odo, axis=0)])


# ---------------------------------------------------------------- fusion
def local_consensus(p, a_idx, ax, ay, half=HALF_WIN, r_agree=R_AGREE, min_support=MIN_SUPPORT,
                    path_s=None, half_m=None, offset_model="loclin", min_span_m=None):
    """Window = anchors within +-half STILLS of still i, or -- when half_m is given -- within +-half_m METRES of
    path distance along the dead-reckoned odometry track (path_s, cumulative, no GPS). The still-count window
    silently changes meaning with the capture rate: +-10 stills is +-800-950 m on UAV-VisLoc but +-34 m on AerialVL
    (2 Hz), where neighbouring anchors share the same wrong pick and outvote the truth (2026-11-23).

    offset_model (2026-11-24, INVESTIGATION_LOG "Fusion-Tail Diagnostic"): how the winning cluster's offsets become
    still i's offset. "mean" (the previous default): their plain mean. "loclin" (default since 2026-11-24, passed its
    pre-registered held-out gate, user decision): a tricube-weighted local-linear fit of
    the offsets against odometry path distance, evaluated AT still i -- the flat mean cannot follow an odometry track
    that bends inside the window (median bend 135-513 m on 01/08/av1 vs 14-33 m anchor noise). Kernel half-width =
    the existing window (no new parameter). One-sided clusters: the line is extrapolated no further than the members'
    own span, else the nearest member's offset is held (see _local_linear_offset)."""
    n = len(p)
    o = np.column_stack([ax - p[a_idx, 0], ay - p[a_idx, 1]])
    est = np.full((n, 2), np.nan)
    for i in range(n):
        if half_m is not None:
            c = np.where(np.abs(path_s[a_idx] - path_s[i]) <= half_m)[0]
        else:
            c = np.where(np.abs(a_idx - i) <= half)[0]
        if len(c) < min_support:
            continue
        d = np.linalg.norm(o[c][:, None] - o[c][None], axis=2)
        votes = (d <= r_agree).sum(axis=1)
        b = int(np.argmax(votes))
        if votes[b] >= min_support:
            members = c[d[b] <= r_agree]
            # Opt-in span guard (2026-11-26, exploratory): a decoy crop hit repeatedly sits at ONE world position, so its
            # offsets agree only while the platform moves < r_agree; correct anchors agree at any separation. A winning
            # cluster spanning < min_span_m of odometry path is therefore not trusted (no consensus -> fill/bridge).
            if min_span_m is not None and np.ptp(path_s[a_idx[members]]) < min_span_m:
                continue
            if offset_model == "mean":
                est[i] = p[i] + o[members].mean(axis=0)
            else:
                est[i] = p[i] + _local_linear_offset(o[members], path_s[a_idx[members]] - path_s[i],
                                                     half_m if half_m is not None
                                                     else np.abs(path_s[a_idx[c]] - path_s[i]).max() + 1.0)
    return est


def _local_linear_offset(off, ds, h):
    """Offset at ds = 0 from members' offsets `off` [m, 2] at signed path distances `ds` [m] (see local_consensus).
    Tricube-weighted (half-width h) least-squares line, evaluated at ds = 0. Most drift stills have a ONE-SIDED cluster
    (01 88%, 08 81%, av1 63%), so the line is also used to extrapolate -- but never further than the members' own span
    (gap from i to the nearest member <= span of the members); beyond that, the nearest member's offset is held."""
    w = np.clip(1.0 - (np.abs(ds) / h) ** 3, 0.0, None) ** 3
    w = np.where(w > 1e-9, w, 1e-9)
    if len(ds) >= 3:
        span = ds.max() - ds.min()
        gap = 0.0 if ds.min() < 0.0 < ds.max() else np.abs(ds).min()
        if gap <= span:
            X = np.column_stack([np.ones_like(ds), ds])
            XtW = X.T * w
            A = XtW @ X
            if abs(np.linalg.det(A)) >= 1e-9:
                return np.linalg.solve(A, XtW @ off)[0]   # intercept = offset at still i
    return off[np.argmin(np.abs(ds))]


# ---------------------------------------------------------------- evaluation
def stats(e):
    e = np.asarray(e)
    if e.size == 0:
        return {k: float("nan") for k in ("mean", "median", "p90", "A@20", "A@50", "A@100")}
    return {"mean": float(e.mean()), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "A@20": float(np.mean(e <= 20)), "A@50": float(np.mean(e <= 50)), "A@100": float(np.mean(e <= 100))}


def fmt(label, s, extra=""):
    return (f"  {label:<30} mean {s['mean']:7.0f}  med {s['median']:7.0f}  p90 {s['p90']:7.0f}  "
            f"A@20 {s['A@20']:5.1%}  A@50 {s['A@50']:5.1%}  A@100 {s['A@100']:5.1%}{extra}")


def gate_and_fill(est, p, a_idx, ax, ay, n, gate_r, fill="forward"):
    """Consensus as a GATE, not an estimator (2026-11-18): where a still's own anchor lies within
    gate_r of the consensus prediction, keep that anchor (correct picks are ~26 m precise, while
    the consensus mean carries odometry drift across the window); else use the consensus. Stills
    with no estimate are filled by dead-reckoning from the nearest covered still (no GT used).
    fill="bridge" (2026-11-22): a gap with covered stills on BOTH sides is dead-reckoned from both
    and blended linearly by distance, so the drift is pinned at both ends; one-sided gaps (flight
    start/end) are unchanged."""
    A = np.full((n, 2), np.nan)
    A[a_idx] = np.column_stack([ax, ay])
    agree = np.linalg.norm(A - est, axis=1) <= gate_r          # NaN compares False
    out = np.where(agree[:, None], A, est)
    cov = np.where(~np.isnan(out[:, 0]))[0]
    if cov.size:
        for i in np.where(np.isnan(out[:, 0]))[0]:
            left, right = cov[cov < i], cov[cov > i]
            if fill == "bridge" and left.size and right.size:
                l, r = left[-1], right[0]
                from_l, from_r = out[l] + (p[i] - p[l]), out[r] + (p[i] - p[r])
                out[i] = ((r - i) * from_l + (i - l) * from_r) / (r - l)
            else:
                j = cov[np.argmin(np.abs(cov - i))]
                out[i] = out[j] + (p[i] - p[j])
    return out


def run_flight(flight, estimator, calib, oracle=False, plot=True, min_inliers=0, r_agree=R_AGREE,
               half=HALF_WIN, gate_r=0.0, prior="speed", fill="forward", half_m=None, offset_model="loclin",
               per_still_out=None, min_span_m=None):
    from batch_trajectory_fusion import batch_fuse   # Step-1 LOESS baseline, unchanged
    S = load_flight(flight, estimator)
    G = np.column_stack([S["gx"], S["gy"]])
    odo = predict_steps(S, calib, prior=prior)
    ok = S["vo_ok"]
    step_err = np.linalg.norm(odo - np.diff(G, axis=0), axis=1)
    # Optional inlier gate (2026-11-18): after the stale query-cache fix, inlier counts separate
    # correct picks from wrong ones (flight 01 AUC 0.80), so low-inlier anchors can be dropped.
    keep = S["a_inl"] >= min_inliers
    a_idx, ax, ay = S["a_idx"][keep], S["ax"][keep], S["ay"][keep]
    raw_err = np.hypot(ax - G[a_idx, 0], ay - G[a_idx, 1])

    fx, fy, _, _ = batch_fuse(S["t"][a_idx].astype(float), ax, ay)
    loess_err = np.hypot(fx - G[a_idx, 0], fy - G[a_idx, 1])

    p = dead_reckon(odo)                                   # arbitrary origin: no GT
    path_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(odo, axis=1))])   # odometry path length, no GT
    est = local_consensus(p, a_idx, ax, ay, half=half, r_agree=r_agree, path_s=path_s, half_m=half_m,
                          offset_model=offset_model, min_span_m=min_span_m)
    if gate_r > 0:
        est = gate_and_fill(est, p, a_idx, ax, ay, len(G), gate_r, fill=fill)
    cov = ~np.isnan(est[:, 0])
    cons_err = np.linalg.norm(est[cov] - G[cov], axis=1)
    if per_still_out:   # opt-in (2026-11-26): per-still fused error, for segment-level scoring; changes nothing else
        vid = [int(fn) - 1 for fn in S["vid_num"]]
        with open(per_still_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Fused_Err_M"])
            for i in range(len(G)):
                w.writerow([vid[i], f"{np.linalg.norm(est[i] - G[i]):.3f}" if cov[i] else ""])

    res = {"flight": flight, "estimator": estimator, "n": len(G), "alt_source": S["alt_source"],
           "vo_ok_frac": float(ok.mean()),
           "vo_step_err_median": float(np.median(step_err[ok])) if ok.any() else float("nan"),
           "anchors_within_150m_frac": float(np.mean(raw_err <= R_AGREE)),
           "raw": stats(raw_err), "loess": stats(loess_err), "consensus": stats(cons_err),
           "consensus_coverage": float(cov.mean())}
    print(f"\n=== flight {flight} ({estimator}) stills={len(G)}  altitude: {S['alt_source']}  "
          f"[min_inliers {min_inliers}: {keep.sum()}/{keep.size} anchors kept, r_agree {r_agree:.0f} m, "
          f"{f'half {half_m:.0f} m (path)' if half_m is not None else f'half {half}'}, "
          f"gate_r {gate_r:.0f}, prior {prior}, fill {fill}"
          f"{', offset ' + offset_model if offset_model != 'mean' else ''}]")
    print(f"  VO ok {res['vo_ok_frac']:.0%} | VO step err median {res['vo_step_err_median']:.1f} m | "
          f"anchors within 150 m: {res['anchors_within_150m_frac']:.1%}")
    print(fmt("raw anchors", res["raw"]))
    print(fmt("Step-1 LOESS (prior best)", res["loess"]))
    print(fmt("stills-VO + local consensus", res["consensus"], f"  (coverage {cov.mean():.0%})"))

    if oracle:   # anchor-selection floor: only anchors truly within 150 m may vote
        good = raw_err <= R_AGREE
        est_o = local_consensus(p, a_idx[good], ax[good], ay[good], r_agree=r_agree, min_support=1)
        cov_o = ~np.isnan(est_o[:, 0])
        res["oracle"] = stats(np.linalg.norm(est_o[cov_o] - G[cov_o], axis=1))
        print(fmt("ORACLE anchor selection", res["oracle"], f"  (coverage {cov_o.mean():.0%})"))

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axp = plt.subplots(figsize=(8, 7))
        axp.plot(G[:, 0], G[:, 1], "-", color="#1f77b4", lw=1.5, label="ground truth")
        axp.scatter(ax, ay, s=5, color="#bbbbbb", label="raw anchors")
        axp.plot(est[:, 0], est[:, 1], "-", color="#d62728", lw=1.2, label="stills-VO + local consensus")
        axp.set_aspect("equal"); axp.legend(fontsize=8); axp.set_xlabel("E (m)"); axp.set_ylabel("N (m)")
        axp.set_title(f"Flight {flight}: stills-VO + {estimator} anchors")
        fig.tight_layout(); fig.savefig(OUT_DIR / f"flight{flight}_{estimator}.png", dpi=120); plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimator", default="split_vlad_disk")
    ap.add_argument("--flights", default=",".join(DEV + VAL),
                    help="default dev+val; held-out (02,05,06,11) only when passed explicitly")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="also report the oracle anchor-selection floor")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--min-span-m", type=float, default=None,
                    help="exploratory guard: ignore a winning consensus cluster spanning < this many m of odometry path")
    ap.add_argument("--per-still-out", default=None,
                    help="write Frame (0-based video index),Fused_Err_M per still; one flight only")
    ap.add_argument("--min-inliers", type=int, default=0, help="drop anchors with fewer inliers")
    ap.add_argument("--r-agree", type=float, default=R_AGREE, help="consensus agreement radius (m)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--half", type=int, default=HALF_WIN, help="consensus window, stills either side")
    ap.add_argument("--half-m", type=float, default=None,
                    help="consensus window in METRES of odometry path either side (overrides --half); "
                         "portable across capture rates. 870 = what --half 10 spans on UAV-VisLoc dev")
    ap.add_argument("--gate-r", type=float, default=0.0,
                    help=">0: keep a still's own anchor when within this many m of consensus, fill gaps")
    ap.add_argument("--prior", choices=("speed", "steplen"), default="speed",
                    help="weak prior for failed VO pairs: recent speed*dt (original) or recent step length")
    ap.add_argument("--fill", choices=("forward", "bridge"), default="forward",
                    help="with --gate-r: fill gaps from the nearest covered still, or bridge from both sides")
    ap.add_argument("--offset-model", choices=("mean", "loclin"), default="loclin",
                    help="consensus offset: local-linear fit along the odometry path evaluated at the still (default "
                         "since 2026-11-24, see local_consensus) or the plain mean of the winning cluster (previous default)")
    a = ap.parse_args()
    flights = a.flights.split(",")
    if a.per_still_out and len(flights) != 1:
        sys.exit("--per-still-out takes exactly one flight")
    if a.oracle and set(flights) & set(HELD_OUT):
        sys.exit("--oracle uses ground truth to pick anchors; not allowed on held-out flights")

    if a.recalibrate or not CALIB_PATH.exists():
        calib = fit_calibration(DEV, a.estimator)
        json.dump(calib, open(CALIB_PATH, "w"), indent=2)
    calib = json.load(open(CALIB_PATH))
    print(f"odometry calibration (fit on {calib['flights']}, {calib['n_pairs']} pairs): residual median "
          f"{calib['fit_resid_median_m']:.1f} m, p90 {calib['fit_resid_p90_m']:.1f} m, implied HFOV "
          f"{calib['implied_hfov_deg']:.1f} deg")

    results = [run_flight(fl, a.estimator, calib, oracle=a.oracle, plot=not a.no_plot,
                          min_inliers=a.min_inliers, r_agree=a.r_agree, half=a.half, gate_r=a.gate_r,
                          prior=a.prior, fill=a.fill, half_m=a.half_m, offset_model=a.offset_model,
                          per_still_out=a.per_still_out, min_span_m=a.min_span_m)
               for fl in flights]
    print(f"\n=== combined over {flights} (mean of per-flight values)")
    for key in ("raw", "loess", "consensus") + (("oracle",) if a.oracle else ()):
        print(fmt(key, {k: float(np.mean([r[key][k] for r in results])) for k in results[0][key]}))
    if a.json_out:
        json.dump(results, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
