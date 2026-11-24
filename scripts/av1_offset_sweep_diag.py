#!/usr/bin/env python3
"""
Follow-up to av1_dusk_matching_diag.py (2026-11-24): how fast do XFeat inliers fall as the reference crop's centre
moves away from the true position, on dark av1 vs daylight av2 frames?

The first diag found dark av1 frames giving 140-225 inliers against a GT-centred crop but only 10-70 against the
best top-10 grid candidate even when a correct (<= 150 m) cell was retrieved. Hypothesis: the binding loss is crop
MIS-CENTRING on the footprint grid (spacing 119 m -> nearest cell up to ~84 m off on a 178 m crop), not darkness.
Untreated query, same geometry/matcher as the first diag. Also reports the nearest grid cell's own offset + inliers
and the max-inlier decoy (> 150 m) among the top-10 retrieved, read from the first diag's per-frame CSV.

    python3 -I scripts/av1_offset_sweep_diag.py
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

OFFSETS_M = (0, 20, 40, 60, 80, 100)
DIRS = [(math.cos(a), math.sin(a)) for a in np.radians(np.arange(0, 360, 45))]


def main():
    bounds = grid.load_bounds("aerialvl_large", str(d.COORDS))
    map_img = cv2.imread(str(d.MAP))
    geo = d.MapGeo(map_img, bounds)
    mpp = grid.map_mpp(map_img, bounds)
    xf = d.XFeat(d.MODELS / "xfeat.onnx")
    per_frame = list(csv.DictReader(open(d.OUT_DIR / "per_frame.csv", newline="")))
    frames = sorted({(r["seq"], int(r["frame"])) for r in per_frame})
    decoy = {(r["seq"], int(r["frame"])): int(r["best_decoy_inl"]) for r in per_frame if r["treat"] == "none"}

    out = []
    for seq, i in frames:
        rows = list(csv.DictReader(open(d.SEQ_DIR / seq / f"{seq}.csv", newline="")))
        r = rows[i]
        gsd = grid.frame_gsd(float(r["height"]), 2048)
        crop_m = int(round(1536 * gsd))
        spacing = int(round(200.0 * crop_m / 300.0))
        glats, glngs = grid.generate_grid(*bounds, grid_spacing_m=float(spacing))
        lat, lng = float(r["lat"]), float(r["lon"])
        frame = cv2.imread(str(d.SEQ_DIR / seq / "drone" / r["filename"]))
        q = grid.make_query(frame, "northup", gsd, mpp, heading_deg=float(r["Phi1"]), crop_size_m=float(crop_m))
        qf = xf.extract(q)
        m_lat = grid.M_PER_DEG_LAT
        m_lng = grid.m_per_deg_lng(lat)
        curve = []
        for off in OFFSETS_M:
            vals = []
            for dx, dy in (DIRS if off else DIRS[:1]):
                c = geo.crop(lat + dy * off / m_lat, lng + dx * off / m_lng, crop_m)
                vals.append(d.match(qf, xf.extract(c[0]))[0] if c is not None else 0)
            curve.append(float(np.median(vals)))
        nc = grid.nearest_grid_index(lat, lng, glats, glngs)
        n_off = geo.err_m(*geo.px(glats[nc], glngs[nc]), lat, lng)
        n_inl = d.match(qf, xf.extract(geo.crop(glats[nc], glngs[nc], crop_m)[0]))[0]
        out.append((seq, i, curve, n_off, n_inl, decoy[(seq, i)]))
        print(f"{seq} {i:4d} | " + " ".join(f"{o}m:{v:5.0f}" for o, v in zip(OFFSETS_M, curve)) +
              f" | nearest cell {n_off:5.1f} m off: {n_inl:4d} inl | best decoy {decoy[(seq, i)]:4d}", flush=True)

    for seq, label in (("av1", "av1 DARK"), ("av2", "av2 DAYLIGHT")):
        R = [x for x in out if x[0] == seq]
        med = np.median(np.array([x[2] for x in R]), axis=0)
        rel = med / max(med[0], 1)
        print(f"\n{label} (n={len(R)}) median inliers by crop-centre offset:")
        print("  " + "  ".join(f"{o:3d} m: {m:5.0f} ({rr * 100:3.0f}%)" for o, m, rr in zip(OFFSETS_M, med, rel)))
        beats = sum(x[4] > x[5] for x in R)
        print(f"  nearest grid cell: median offset {np.median([x[3] for x in R]):.0f} m, median inliers "
              f"{np.median([x[4] for x in R]):.0f}; beats best top-10 decoy on {beats}/{len(R)} frames; "
              f"median best decoy {np.median([x[5] for x in R]):.0f}")


if __name__ == "__main__":
    main()
