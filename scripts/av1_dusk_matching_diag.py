#!/usr/bin/env python3
"""
av1 dusk matching diagnostic: oracle XFeat inliers on dark frames, with photometric query treatments (2026-11-24).

Context (INVESTIGATION_LOG "AerialVL transfer test"): on AerialVL av1 (dusk), retrieval survives (R@10 50-69%) but
XFeat matching loses discrimination; frames with mean brightness < 35/255 reach raw A@50 ~6%. This script asks
whether a cheap photometric treatment of the QUERY (drone) side restores XFeat's ability to pick the right crop, and
checks the same treatments on daylight av2 frames (must not hurt).

Held fixed (single variable = the photometric treatment applied before XFeat):
  - query geometry: C++-mirroring north-up crop (uavvisloc_grid.make_query, 'northup', heading = Phi1, GSD from the
    C++ altitude/HFOV 51.5), crop = footprint (the default footprint-matched geometry: av1 178 m / 119 m spacing,
    av2 200 m / 133 m);
  - retrieval: fine-tuned dense DINOv2 + VLAD k=64 on the UNTREATED query, top-10 over the footprint grid
    (mirrors split_vlad_xfeat; retrieval is not the variable here);
  - matching: XFeat ONNX, score >= 0.05, mutual-NN + cosine >= 0.82, findHomography RANSAC 5 px (XFeatMatching.cpp).

Treatments (all computed on the channel-mean grey image XFeat itself sees; 'none' passes the original BGR):
  none       untouched
  gain       linear stretch to the map's grey mean/std. CONTROL: XFeat's first op is InstanceNorm on that grey
             image, so this should be ~null (only 8-bit rounding differs). A non-null result means a bug here.
  gamma      per-frame gamma so the query's grey mean -> the map's grey mean (non-linear: lifts shadows)
  clahe_q    CLAHE (clip 2.0, 8x8 tiles) on the query only
  clahe_both CLAHE on query AND reference crops (symmetric)
  histmatch  query grey histogram matched to the whole map's grey histogram (one CDF, candidate-independent)

Per frame and treatment: inliers vs the GT-centred crop (oracle matchability) and its homography position error;
then over the top-10 retrieved candidates: the pick (max inliers), its homography-projected error (raw A@50), and
best-correct (centre <= 150 m from GT, the R@k radius) vs best-decoy inliers.

    source ../CV_IP/cv_env/bin/activate
    python3 -I scripts/av1_dusk_matching_diag.py [--n-dark 24] [--n-day 12]
"""
import argparse
import csv
import math
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import proto_vlad_retrieval as pv  # noqa: E402
import uavvisloc_grid as grid  # noqa: E402

COORDS = REPO / "Datasets" / "aerialvl_coordinates_range.csv"
MAP = REPO / "Images" / "map_clean_aerialvl_large.png"
SEQ_DIR = REPO / "Datasets" / "UAV_VisLoc_dataset"
MODELS = REPO / "models"
OUT_DIR = REPO / "Images" / "av1_dusk_diag"
TREATMENTS = ("none", "gain", "gamma", "clahe_q", "clahe_both", "histmatch")
CORRECT_RADIUS_M = 150.0
TOP_K = 10

# XFeatMatching.cpp constants
XF_INPUT, XF_SCORE, XF_MIN_KP, XF_MIN_COS, XF_RANSAC = 384, 0.05, 150, 0.82, 5.0


def read_vocab(path):
    raw = Path(path).read_bytes()
    magic, k, d = struct.unpack("<iii", raw[:12])
    assert magic == 0x44414C56, path
    return np.frombuffer(raw[12:], dtype="<f4").reshape(k, d).astype(np.float32)


class XFeat:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def extract(self, bgr):
        h, w = bgr.shape[:2]
        x = cv2.resize(bgr, (XF_INPUT, XF_INPUT)).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        kp, desc, sc = self.s.run(["keypoints", "descriptors", "scores"], {"image": x})
        keep = sc[0] >= XF_SCORE
        kp = kp[0][keep] * np.array([w / XF_INPUT, h / XF_INPUT], np.float32)
        desc = desc[0][keep].astype(np.float32)
        return (kp, desc) if len(kp) >= XF_MIN_KP else None


def match(qf, rf):
    """Mutual-NN + cosine threshold + RANSAC homography; returns (inliers, H) or (0, None)."""
    if qf is None or rf is None:
        return 0, None
    sim = qf[1] @ rf[1].T
    a = sim.argmax(1)
    b = sim.argmax(0)
    qi = np.where((b[a] == np.arange(len(a))) & (sim[np.arange(len(a)), a] > XF_MIN_COS))[0]
    if len(qi) < 8:
        return 0, None
    src = qf[0][qi].reshape(-1, 1, 2)
    dst = rf[0][a[qi]].reshape(-1, 1, 2)
    H, m = cv2.findHomography(src, dst, cv2.RANSAC, XF_RANSAC)
    if H is None or m is None:
        return 0, None
    return int(m.sum()), H


# ---- photometric treatments (on the channel-mean grey, as XFeat sees it) ----
def grey(bgr):
    return np.clip(np.rint(bgr.astype(np.float32).mean(axis=2)), 0, 255).astype(np.uint8)


def as3(g):
    return cv2.merge([g, g, g])


CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def treat(bgr, name, map_stats):
    if name == "none":
        return bgr
    g = grey(bgr)
    if name == "gain":
        m, s = g.mean(), g.std() + 1e-6
        out = (g.astype(np.float32) - m) / s * map_stats["std"] + map_stats["mean"]
        return as3(np.clip(np.rint(out), 0, 255).astype(np.uint8))
    if name == "gamma":
        m = max(g.mean(), 1.0) / 255.0
        gam = math.log(map_stats["mean"] / 255.0) / math.log(m)
        lut = np.clip(np.rint(255.0 * (np.arange(256) / 255.0) ** gam), 0, 255).astype(np.uint8)
        return as3(cv2.LUT(g, lut))
    if name in ("clahe_q", "clahe_both"):
        return as3(CLAHE.apply(g))
    if name == "histmatch":
        hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
        cdf = np.cumsum(hist) / hist.sum()
        lut = np.clip(np.searchsorted(map_stats["cdf"], cdf), 0, 255).astype(np.uint8)
        return as3(cv2.LUT(g, lut))
    raise ValueError(name)


# ---- map geometry ----
class MapGeo:
    def __init__(self, img, bounds):
        self.img, self.b = img, bounds
        lt_lat, lt_lon, rb_lat, rb_lon = bounds
        h, w = img.shape[:2]
        self.mpp_x = (rb_lon - lt_lon) * grid.m_per_deg_lng((lt_lat + rb_lat) / 2) / w
        self.mpp_y = (lt_lat - rb_lat) * grid.M_PER_DEG_LAT / h

    def px(self, lat, lng):
        lt_lat, lt_lon, rb_lat, rb_lon = self.b
        h, w = self.img.shape[:2]
        return (lng - lt_lon) / (rb_lon - lt_lon) * w, (lat - lt_lat) / (rb_lat - lt_lat) * h

    def crop(self, lat, lng, size_m):
        """Returns (crop, x0, y0) like grid.crop_at, plus its top-left pixel."""
        h, w = self.img.shape[:2]
        hx, hy = int(round(size_m / self.mpp_x / 2)), int(round(size_m / self.mpp_y / 2))
        px, py = self.px(lat, lng)
        px, py = int(round(px)), int(round(py))
        x0, x1, y0, y1 = max(0, px - hx), min(w, px + hx), max(0, py - hy), min(h, py + hy)
        if x1 - x0 < 2 * hx * 0.5 or y1 - y0 < 2 * hy * 0.5:
            return None
        return self.img[y0:y1, x0:x1], x0, y0

    def err_m(self, map_px, map_py, lat, lng):
        gx, gy = self.px(lat, lng)
        return math.hypot((map_px - gx) * self.mpp_x, (map_py - gy) * self.mpp_y)


def projected_error(H, q_shape, x0, y0, geo, lat, lng):
    if H is None:
        return float("nan")
    c = np.array([[[q_shape[1] / 2.0, q_shape[0] / 2.0]]], np.float32)
    p = cv2.perspectiveTransform(c, H)[0, 0]
    return geo.err_m(x0 + p[0], y0 + p[1], lat, lng)


def frame_brightness(path):
    return float(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dark", type=int, default=24, help="av1 frames with brightness < 35, evenly spaced")
    ap.add_argument("--n-day", type=int, default=12, help="av2 daylight control frames (100-449), evenly spaced")
    ap.add_argument("--cache", default=None, help="dir for reference-embedding caches (default Images/av1_dusk_diag)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache) if args.cache else OUT_DIR

    bounds = grid.load_bounds("aerialvl_large", str(COORDS))
    map_img = cv2.imread(str(MAP))
    geo = MapGeo(map_img, bounds)
    mpp = grid.map_mpp(map_img, bounds)
    mg = grey(map_img)
    hist = np.bincount(mg.ravel(), minlength=256).astype(np.float64)
    map_stats = {"mean": float(mg.mean()), "std": float(mg.std()), "cdf": np.cumsum(hist) / hist.sum()}
    print(f"map mpp {mpp:.3f}; map grey mean/std {map_stats['mean']:.1f}/{map_stats['std']:.1f}")

    xf = XFeat(MODELS / "xfeat.onnx")
    emb = pv.DenseEmbedder(str(MODELS / "dinov2_s_finetuned_dense.onnx"))
    vocab = read_vocab(MODELS / "vlad_vocab_finetuned_k64.bin")

    seqs = {}
    for seq in ("av1", "av2"):
        rows = list(csv.DictReader(open(SEQ_DIR / seq / f"{seq}.csv", newline="")))
        alt = float(rows[0]["height"])
        gsd = grid.frame_gsd(alt, 2048)
        foot = 1536 * gsd
        crop_m = int(round(foot))                       # footprintMatchedCropGeometry (ratio < 0.75 on both)
        spacing = int(round(200.0 * crop_m / 300.0))
        glats, glngs = grid.generate_grid(*bounds, grid_spacing_m=float(spacing))
        seqs[seq] = dict(rows=rows, gsd=gsd, crop_m=crop_m, spacing=spacing, glats=glats, glngs=glngs)
        print(f"{seq}: GSD {gsd:.4f} m/px, footprint {foot:.1f} m -> crop {crop_m} m / spacing {spacing} m, "
              f"{len(glats)} grid cells")

    # ---- frame selection (brightness computed on the full frame, as in the log) ----
    bright_cache = cache_dir / "av1_brightness.npy"
    if bright_cache.exists():
        b1 = np.load(bright_cache)
    else:
        b1 = np.array([frame_brightness(SEQ_DIR / "av1" / "drone" / r["filename"]) for r in seqs["av1"]["rows"]])
        np.save(bright_cache, b1)
    dark = np.where(b1 < 35.0)[0]
    pick_dark = dark[np.linspace(0, len(dark) - 1, args.n_dark).round().astype(int)]
    pick_day = np.linspace(100, 449, args.n_day).round().astype(int)
    print(f"av1 brightness range {b1.min():.1f}-{b1.max():.1f}; {len(dark)} frames < 35; "
          f"chosen dark frames {pick_dark.tolist()}")
    print(f"av2 daylight control frames {pick_day.tolist()}\n")

    # ---- C++ reference: the committed av1/av2 fp telemetry for the same frames ----
    cpp = {}
    for seq in ("av1", "av2"):
        p = REPO / "CSV Files" / f"video_telemetry_split_vlad_xfeat_fp_uavvisloc_uavvisloc_{seq}.csv"
        cpp[seq] = {int(r["Frame"]): r for r in csv.DictReader(open(p, newline=""))}

    results = []
    montage_rows = []
    for seq, frames in (("av1", pick_dark), ("av2", pick_day)):
        S = seqs[seq]
        # reference VLAD embeddings for this grid (cached)
        ec = cache_dir / f"refvlad_{seq}_c{S['crop_m']}_s{S['spacing']}.npy"
        crops = [geo.crop(la, ln, S["crop_m"]) for la, ln in zip(S["glats"], S["glngs"])]
        if ec.exists():
            refv = np.load(ec)
        else:
            print(f"embedding {len(crops)} {seq} reference crops (one-time, cached to {ec}) ...", flush=True)
            refv = np.stack([pv.vlad_encode(emb.embed_dense(c[0]), vocab) if c is not None else
                             np.zeros(vocab.size, np.float32) for c in crops])
            np.save(ec, refv)
        ref_feat = {}  # (cell, treated?) -> features

        def rfeat(cell, clahe):
            key = (cell, clahe)
            if key not in ref_feat:
                img = crops[cell][0]
                ref_feat[key] = xf.extract(as3(CLAHE.apply(grey(img))) if clahe else img)
            return ref_feat[key]

        for i in frames:
            r = S["rows"][i]
            lat, lng = float(r["lat"]), float(r["lon"])
            frame = cv2.imread(str(SEQ_DIR / seq / "drone" / r["filename"]))
            q = grid.make_query(frame, "northup", S["gsd"], mpp, heading_deg=float(r["Phi1"]),
                                crop_size_m=float(S["crop_m"]))
            qv = pv.vlad_encode(emb.embed_dense(q), vocab)
            sims = refv @ qv
            top = np.argsort(-sims)[:TOP_K]
            dist = [geo.err_m(*geo.px(S["glats"][c], S["glngs"][c]), lat, lng) for c in top]
            correct = [d <= CORRECT_RADIUS_M for d in dist]
            rank = next((k for k, ok in enumerate(correct) if ok), -1)
            oc = geo.crop(lat, lng, S["crop_m"])
            c_row = cpp[seq].get(i, {})
            cpp_err = float("nan")
            if c_row:
                cpp_err = geo.err_m(*geo.px(float(c_row["Raw_Lat"]), float(c_row["Raw_Lng"])), lat, lng)
            for t in TREATMENTS:
                qt = treat(q, t, map_stats)
                qf = xf.extract(qt)
                clahe_ref = t == "clahe_both"
                orc_feat = xf.extract(as3(CLAHE.apply(grey(oc[0]))) if clahe_ref else oc[0])
                o_inl, o_H = match(qf, orc_feat)
                o_err = projected_error(o_H, qt.shape, oc[1], oc[2], geo, lat, lng)
                cand = []
                for c in top:
                    inl, H = match(qf, rfeat(int(c), clahe_ref))
                    cand.append((inl, H, int(c)))
                best = max(range(TOP_K), key=lambda k: cand[k][0])
                inl_b, H_b, cell_b = cand[best]
                cr = crops[cell_b]
                pick_err = projected_error(H_b, qt.shape, cr[1], cr[2], geo, lat, lng) if inl_b > 0 else float("nan")
                bc = max([cand[k][0] for k in range(TOP_K) if correct[k]], default=-1)
                bd = max([cand[k][0] for k in range(TOP_K) if not correct[k]], default=-1)
                results.append(dict(seq=seq, frame=i, bright=round(float(grey(frame).mean()), 1),
                                    q_bright=round(float(grey(q).mean()), 1), rank=rank, treat=t,
                                    q_kp=0 if qf is None else len(qf[0]), oracle_inl=o_inl,
                                    oracle_err=round(o_err, 1), pick_inl=inl_b, pick_err=round(pick_err, 1),
                                    best_correct_inl=bc, best_decoy_inl=bd,
                                    cpp_inliers=c_row.get("Inliers", ""), cpp_err=round(cpp_err, 1)))
            if len(montage_rows) < 6 and seq == "av1" and i in pick_dark[:: max(1, len(pick_dark) // 6)]:
                tiles = [cv2.resize(treat(q, t, map_stats), (192, 192)) for t in TREATMENTS]
                tiles.append(cv2.resize(oc[0], (192, 192)))
                montage_rows.append(np.hstack(tiles))
            last = [x for x in results if x["frame"] == i and x["seq"] == seq]
            print(f"{seq} {i:4d} bright {last[0]['bright']:5.1f} rank {rank:2d} | " + " ".join(
                f"{x['treat']}:{x['oracle_inl']}/{x['pick_inl']}" for x in last), flush=True)

    out_csv = OUT_DIR / "per_frame.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    if montage_rows:
        hdr = np.full((24, montage_rows[0].shape[1], 3), 255, np.uint8)
        for k, name in enumerate(TREATMENTS + ("GT map crop",)):
            cv2.putText(hdr, name, (k * 192 + 4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(OUT_DIR / "montage_dark_av1.png"), np.vstack([hdr] + montage_rows))

    # ---- summary ----
    def med(v):
        v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
        return float(np.median(v)) if v else float("nan")

    print(f"\nper-frame CSV: {out_csv}")
    for seq, label in (("av1", "av1 DARK (< 35)"), ("av2", "av2 DAYLIGHT control")):
        print(f"\n=== {label} ===")
        print(f"{'treatment':11s} {'q_kp':>5s} {'oracle inl':>10s} {'oracle err':>10s} {'oracle<50m':>10s} "
              f"{'pick A@50':>9s} {'pick med err':>12s} {'correct>decoy':>13s} (of frames w/ correct in top-10)")
        for t in TREATMENTS:
            R = [x for x in results if x["seq"] == seq and x["treat"] == t]
            n = len(R)
            inr = [x for x in R if x["rank"] >= 0]
            win = sum(x["best_correct_inl"] > x["best_decoy_inl"] for x in inr)
            o50 = sum(x["oracle_err"] < 50 for x in R if not math.isnan(x["oracle_err"]))
            a50 = sum(x["pick_err"] < 50 for x in R if not math.isnan(x["pick_err"]))
            print(f"{t:11s} {med([x['q_kp'] for x in R]):5.0f} {med([x['oracle_inl'] for x in R]):10.1f} "
                  f"{med([x['oracle_err'] for x in R]):10.1f} {o50:5d}/{n:<4d} {a50:4d}/{n:<4d} "
                  f"{med([x['pick_err'] for x in R]):12.1f} {win:7d}/{len(inr)}")
        R = [x for x in results if x["seq"] == seq and x["treat"] == "none"]
        c50 = sum(x["cpp_err"] < 50 for x in R if not math.isnan(x["cpp_err"]))
        print(f"C++ committed run, same frames: raw A@50 {c50}/{len(R)}, median raw err {med([x['cpp_err'] for x in R]):.1f} m"
              f" (cross-check for 'none')")


if __name__ == "__main__":
    main()
