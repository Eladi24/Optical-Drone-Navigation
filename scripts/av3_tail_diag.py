#!/usr/bin/env python3
"""
av3 tail diagnostic (2026-11-26, a declared SECOND LOOK at fresh-test av3; INVESTIGATION_LOG "av3 Tail Diagnostic"):
is the last-quarter failure (frames 806-920: raw A@50 0%, R@10 1%) rotation, unmatchable content, or a retrieval miss?
Same oracle angle sweep as aerialvl_heading_sign_diag.py; nothing is chosen from this.

    python3 -I scripts/av3_tail_diag.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import av1_dusk_matching_diag as d  # noqa: E402
import aerialvl_heading_sign_diag as h  # noqa: E402
import uavvisloc_grid as grid  # noqa: E402

OUT = d.REPO / "Images" / "av3_tail_diag"
SEGMENTS = {"control": range(576, 691, 10), "tail": range(806, 921, 5)}


def classify(peak, at0, dstar, rank):
    if peak < 40:
        return "UNMATCHABLE"
    if peak >= 60 and abs(dstar) >= 15 and peak >= 2 * max(at0, 1):
        return "ROTATION"
    if at0 >= 60 and rank < 0:
        return "RETRIEVAL_MISS"
    return "MIXED" if rank < 0 else "OK"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = grid.load_bounds("aerialvl_large", str(d.COORDS))
    map_img = cv2.imread(str(d.MAP))
    geo = d.MapGeo(map_img, bounds)
    mpp = grid.map_mpp(map_img, bounds)
    xf = d.XFeat(d.MODELS / "xfeat.onnx")
    rows = list(csv.DictReader(open(d.SEQ_DIR / "av3" / "av3.csv", newline="")))
    tel = {int(r["Frame"]): r for r in csv.DictReader(open(
        d.REPO / "CSV Files" / "video_telemetry_split_vlad_xfeat_fp_rc_uavvisloc_uavvisloc_av3.csv", newline=""))}
    cum = h.cum_vo("av3", len(rows))
    gsd = grid.frame_gsd(float(rows[0]["height"]), 2048)
    crop_m = int(round(1536 * gsd))
    out = []
    for seg, frames in SEGMENTS.items():
        for i in frames:
            r = rows[i]
            oc = geo.crop(float(r["lat"]), float(r["lon"]), crop_m)
            rf = xf.extract(oc[0])
            frame = cv2.imread(str(d.SEQ_DIR / "av3" / "drone" / r["filename"]))
            inl = np.array([d.match(xf.extract(grid.make_query(frame, "northup", gsd, mpp, heading_deg=90.0 + float(x),
                                                               crop_size_m=float(crop_m))), rf)[0]
                            for x in h.DELTAS], float)
            dstar, peak = h.peak(h.DELTAS, inl)
            at0 = int(inl[h.DELTAS == 0][0])
            t = tel.get(i, {})
            rank = int(t.get("Correct_Candidate_Rank", -1) or -1)
            cls = classify(peak, at0, dstar, rank)
            out.append(dict(segment=seg, frame=i, cum_vo=round(float(cum[i]), 1), delta_star=round(dstar, 1),
                            peak_inl=peak, inl_at_0=at0, rank=rank, cpp_inliers=t.get("Inliers", ""),
                            brightness=round(float(frame.mean()), 1), cls=cls))
            print(f"{seg:7s} {i:4d} cumVO {cum[i]:6.1f} | delta* {dstar:6.1f} peak {peak:4d} at0 {at0:4d} | "
                  f"rank {rank:2d} cpp_inl {t.get('Inliers', ''):>4s} | {cls}", flush=True)
            if seg == "tail" and i in (806, 851, 896, 916):
                cv2.imwrite(str(OUT / f"tail_{i}.png"), np.hstack([
                    cv2.resize(grid.make_query(frame, "northup", gsd, mpp, heading_deg=90.0 + dstar,
                                               crop_size_m=float(crop_m)), (400, 400)),
                    cv2.resize(oc[0], (400, 400))]))
    with open(OUT / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for seg in SEGMENTS:
        R = [x for x in out if x["segment"] == seg]
        counts = {c: sum(x["cls"] == c for x in R) for c in ("OK", "ROTATION", "UNMATCHABLE", "RETRIEVAL_MISS", "MIXED")}
        print(f"\n{seg}: n={len(R)} {counts} | median peak {np.median([x['peak_inl'] for x in R]):.0f}, "
              f"at0 {np.median([x['inl_at_0'] for x in R]):.0f}, |delta*| {np.median([abs(x['delta_star']) for x in R]):.1f}")


if __name__ == "__main__":
    main()
