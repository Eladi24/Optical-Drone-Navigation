#!/usr/bin/env python3
"""
AerialVL heading sign sweep (2026-11-26): which sign turns the camera-only VO rotation into a pre-warp heading?

Context (INVESTIGATION_LOG "AerialVL fresh test"): AerialVL frames are only approximately image-top = east (Phi1 = 90
constant, no per-frame yaw on disk). Where the camera's real orientation drifts (av2 rotated tail, av3 last quarter),
retrieval/XFeat fail. Candidate fix: Phi1_i = 90 + s * cumVO_i, cumVO_i = sum of frame-to-frame Rot_Deg from
CSV Files/stills_vo_uavvisloc_<seq>.csv (camera-only, no GPS). This script measures s and whether cumVO tracks the true
heading error at all (handoff trap 2: av4's cumVO reaches 64 deg while its retrieval stayed fine).

Per frame: query = C++-mirroring north-up crop (uavvisloc_grid.make_query, footprint crop size) with heading
90 + delta, delta swept over DELTAS; XFeat inliers vs the GT-centred map crop (oracle, retrieval not involved).
delta* = inlier-peak (parabolic refinement) = the frame's true heading offset from 90, measured from the image + GPS.
Then fit delta* = s * cumVO + b over frames with a clear peak.

Design data only: av2 (whole flight, dense on the rotated tail) and av1. av3/av4 are NOT read.
Prediction written before the run: s = -1 (a clockwise camera yaw rotates the world counter-clockwise in the image,
i.e. negative atan2(A10, A00) in y-down coordinates, so heading = 90 - cumVO).

    source ../CV_IP/cv_env/bin/activate
    python3 -I scripts/aerialvl_heading_sign_diag.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import av1_dusk_matching_diag as d  # noqa: E402
import uavvisloc_grid as grid  # noqa: E402

REPO = d.REPO
OUT_DIR = REPO / "Images" / "heading_sign_diag"
DELTAS = np.arange(-72, 73, 3)
MIN_PEAK_INL = 40          # a peak below this is "no clear peak" (decoy level on these flights)


def cum_vo(seq, n_frames):
    """cum[i] = sum of Rot_Deg over consecutive pairs up to 0-based frame i (filename number i+1); failed pairs add 0."""
    rot = np.zeros(n_frames)
    for r in csv.DictReader(open(REPO / "CSV Files" / f"stills_vo_uavvisloc_{seq}.csv", newline="")):
        if r["From_Num"].startswith("#"):
            continue
        to = int(r["To_Num"]) - 1
        if r["Ok"] == "1" and int(r["To_Num"]) == int(r["From_Num"]) + 1 and 0 <= to < n_frames:
            rot[to] = float(r["Rot_Deg"])
    return np.cumsum(rot)


def peak(deltas, inl):
    k = int(np.argmax(inl))
    if 0 < k < len(inl) - 1:
        y0, y1, y2 = inl[k - 1], inl[k], inl[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(deltas[k] + 0.5 * (y0 - y2) / den * (deltas[1] - deltas[0])), int(y1)
    return float(deltas[k]), int(inl[k])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bounds = grid.load_bounds("aerialvl_large", str(d.COORDS))
    map_img = cv2.imread(str(d.MAP))
    geo = d.MapGeo(map_img, bounds)
    mpp = grid.map_mpp(map_img, bounds)
    xf = d.XFeat(d.MODELS / "xfeat.onnx")

    plan = {}
    rows2 = list(csv.DictReader(open(d.SEQ_DIR / "av2" / "av2.csv", newline="")))
    rows1 = list(csv.DictReader(open(d.SEQ_DIR / "av1" / "av1.csv", newline="")))
    plan["av2"] = (rows2, sorted(set(np.linspace(0, 440, 16).round().astype(int)) |
                                 set(range(446, len(rows2), 5))))
    plan["av1"] = (rows1, sorted(set(np.linspace(0, len(rows1) - 1, 16).round().astype(int))))

    out = []
    for seq, (rows, frames) in plan.items():
        cum = cum_vo(seq, len(rows))
        gsd = grid.frame_gsd(float(rows[0]["height"]), 2048)
        crop_m = int(round(1536 * gsd))
        print(f"{seq}: GSD {gsd:.4f}, crop {crop_m} m, cumVO range {cum.min():.1f}..{cum.max():.1f} deg, "
              f"{len(frames)} frames", flush=True)
        for i in frames:
            r = rows[i]
            lat, lng = float(r["lat"]), float(r["lon"])
            oc = geo.crop(lat, lng, crop_m)
            if oc is None:
                continue
            rf = xf.extract(oc[0])
            frame = cv2.imread(str(d.SEQ_DIR / seq / "drone" / r["filename"]))
            inl = []
            for dl in DELTAS:
                q = grid.make_query(frame, "northup", gsd, mpp, heading_deg=float(r["Phi1"]) + float(dl),
                                    crop_size_m=float(crop_m))
                inl.append(d.match(xf.extract(q), rf)[0])
            inl = np.array(inl, float)
            dstar, pinl = peak(DELTAS, inl)
            at0 = int(inl[DELTAS == 0][0])
            out.append(dict(seq=seq, frame=i, cum_vo=round(float(cum[i]), 2), delta_star=round(dstar, 1),
                            peak_inl=pinl, inl_at_0=at0, clear=int(pinl >= MIN_PEAK_INL),
                            curve=" ".join(str(int(v)) for v in inl)))
            print(f"{seq} {i:4d} cumVO {cum[i]:7.1f} | delta* {dstar:6.1f} (peak {pinl:4d} inl, at 0: {at0:4d})",
                  flush=True)

    with open(OUT_DIR / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"\nper-frame CSV: {OUT_DIR / 'per_frame.csv'}  (deltas {DELTAS[0]}..{DELTAS[-1]} step {DELTAS[1]-DELTAS[0]})")
    for seq in ("av2", "av1", "both"):
        R = [x for x in out if x["clear"] and (seq == "both" or x["seq"] == seq)]
        if len(R) < 3:
            print(f"{seq}: too few clear peaks ({len(R)})")
            continue
        c = np.array([x["cum_vo"] for x in R])
        ds = np.array([x["delta_star"] for x in R])
        s, b = np.polyfit(c, ds, 1)
        res_fit = ds - (s * c + b)
        print(f"\n{seq}: {len(R)} clear peaks (of {sum(1 for x in out if seq == 'both' or x['seq'] == seq)}); "
              f"fit delta* = {s:+.3f} * cumVO {b:+.1f}; residual RMS {np.sqrt(np.mean(res_fit**2)):.1f} deg")
        for sgn in (-1, +1):
            e = ds - (sgn * c + np.median(ds - sgn * c))
            print(f"  s = {sgn:+d} (best offset {np.median(ds - sgn * c):+.1f}): |delta* - pred| median "
                  f"{np.median(np.abs(e)):.1f}, max {np.abs(e).max():.1f} deg")
        print(f"  no correction (delta = 0): |delta*| median {np.median(np.abs(ds)):.1f}, max {np.abs(ds).max():.1f}")


if __name__ == "__main__":
    main()
