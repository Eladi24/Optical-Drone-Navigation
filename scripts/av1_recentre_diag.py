#!/usr/bin/env python3
"""
Mechanism check for candidate RE-CENTRING (2026-11-24), on the same 24 dark av1 + 12 daylight av2 frames as
av1_dusk_matching_diag.py. Not a tuning loop and not the gate: the gate is a pre-registered C++ variant on full flights.

Finding being tested: XFeat inliers fall ~44% at 40 m and ~70% at 60 m crop-centre offset, day and dusk alike; the
footprint grid's nearest cell sits a median ~60 m off, which on dark frames (3x fewer inliers) drops the correct cell
to the decoy level (16 vs 14). Re-centring: for every top-10 candidate with a homography, project the query centre
into the map, cut a NEW footprint crop centred there, re-match, and rank candidates by the re-centred inliers (a
candidate whose re-match fails keeps 0). Decoys have no true correspondence, so re-centring should not inflate them
systematically; the correct cell should gain (offset ~60 m -> ~10-20 m). `--passes 2` re-centres twice.

    python3 -I scripts/av1_recentre_diag.py [--passes 1]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import av1_dusk_matching_diag as d  # noqa: E402
import proto_vlad_retrieval as pv  # noqa: E402
import uavvisloc_grid as grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=1)
    args = ap.parse_args()
    bounds = grid.load_bounds("aerialvl_large", str(d.COORDS))
    map_img = cv2.imread(str(d.MAP))
    geo = d.MapGeo(map_img, bounds)
    mpp = grid.map_mpp(map_img, bounds)
    xf = d.XFeat(d.MODELS / "xfeat.onnx")
    emb = pv.DenseEmbedder(str(d.MODELS / "dinov2_s_finetuned_dense.onnx"))
    vocab = d.read_vocab(d.MODELS / "vlad_vocab_finetuned_k64.bin")
    per_frame = list(csv.DictReader(open(d.OUT_DIR / "per_frame.csv", newline="")))
    frames = sorted({(r["seq"], int(r["frame"])) for r in per_frame})

    res = []
    for seq, i in frames:
        rows = list(csv.DictReader(open(d.SEQ_DIR / seq / f"{seq}.csv", newline="")))
        r = rows[i]
        gsd = grid.frame_gsd(float(r["height"]), 2048)
        crop_m = int(round(1536 * gsd))
        spacing = int(round(200.0 * crop_m / 300.0))
        glats, glngs = grid.generate_grid(*bounds, grid_spacing_m=float(spacing))
        refv = np.load(d.OUT_DIR / f"refvlad_{seq}_c{crop_m}_s{spacing}.npy")
        lat, lng = float(r["lat"]), float(r["lon"])
        frame = cv2.imread(str(d.SEQ_DIR / seq / "drone" / r["filename"]))
        q = grid.make_query(frame, "northup", gsd, mpp, heading_deg=float(r["Phi1"]), crop_size_m=float(crop_m))
        qf = xf.extract(q)
        top = np.argsort(-(refv @ pv.vlad_encode(emb.embed_dense(q), vocab)))[:d.TOP_K]
        qc = np.array([[[q.shape[1] / 2.0, q.shape[0] / 2.0]]], np.float32)
        base, rec = [], []
        for c in top:
            cr = geo.crop(glats[c], glngs[c], crop_m)
            correct = geo.err_m(*geo.px(glats[c], glngs[c]), lat, lng) <= d.CORRECT_RADIUS_M
            inl, H = d.match(qf, xf.extract(cr[0]))
            err = d.projected_error(H, q.shape, cr[1], cr[2], geo, lat, lng) if inl else float("nan")
            base.append((inl, err, correct))
            r_inl, r_err, x0, y0, Hc = inl, err, cr[1], cr[2], H
            for _ in range(args.passes):
                if not r_inl:
                    break
                p = cv2.perspectiveTransform(qc, Hc)[0, 0]
                mx, my = x0 + p[0], y0 + p[1]
                lt_lat, lt_lon, rb_lat, rb_lon = bounds
                h, w = map_img.shape[:2]
                nlat = lt_lat + my / h * (rb_lat - lt_lat)
                nlng = lt_lon + mx / w * (rb_lon - lt_lon)
                nc = geo.crop(nlat, nlng, crop_m)
                if nc is None:
                    r_inl = 0
                    break
                r_inl, Hc = d.match(qf, xf.extract(nc[0]))
                x0, y0 = nc[1], nc[2]
                r_err = d.projected_error(Hc, q.shape, x0, y0, geo, lat, lng) if r_inl else float("nan")
            rec.append((r_inl, r_err, correct))
        row = {"seq": seq, "frame": i}
        for tag, cand in (("base", base), ("recentred", rec)):
            b = max(range(len(cand)), key=lambda k: cand[k][0])
            row[f"{tag}_err"] = cand[b][1]
            row[f"{tag}_inl"] = cand[b][0]
            row[f"{tag}_bc"] = max([x[0] for x in cand if x[2]], default=-1)
            row[f"{tag}_bd"] = max([x[0] for x in cand if not x[2]], default=-1)
        row["has_correct"] = any(x[2] for x in base)
        res.append(row)
        print(f"{seq} {i:4d} correct-in-top10 {int(row['has_correct'])} | base pick {row['base_inl']:4d} inl "
              f"{row['base_err']:7.1f} m (best correct {row['base_bc']:4d} / decoy {row['base_bd']:4d}) | "
              f"recentred pick {row['recentred_inl']:4d} inl {row['recentred_err']:7.1f} m "
              f"(best correct {row['recentred_bc']:4d} / decoy {row['recentred_bd']:4d})", flush=True)

    for seq, label in (("av1", "av1 DARK"), ("av2", "av2 DAYLIGHT")):
        R = [x for x in res if x["seq"] == seq]
        H = [x for x in R if x["has_correct"]]
        print(f"\n{label} (n={len(R)}, {len(H)} with a correct cell in the top 10), passes={args.passes}")
        for tag in ("base", "recentred"):
            e = np.array([x[f"{tag}_err"] for x in R], float)
            a50 = int(np.sum(e < 50))
            a20 = int(np.sum(e < 20))
            win = sum(x[f"{tag}_bc"] > x[f"{tag}_bd"] for x in H)
            print(f"  {tag:10s} pick A@20 {a20:2d}/{len(R)}  A@50 {a50:2d}/{len(R)}  median err "
                  f"{np.nanmedian(e):7.1f} m | correct>decoy {win:2d}/{len(H)} | median best-correct inl "
                  f"{np.median([x[f'{tag}_bc'] for x in H]):5.1f}, best-decoy {np.median([x[f'{tag}_bd'] for x in R]):5.1f}")


if __name__ == "__main__":
    main()
