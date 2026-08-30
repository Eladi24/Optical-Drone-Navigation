#!/usr/bin/env python3
"""
Cheap Python prototype for STRATEGY.md's Phase 2+ dense/semantic matching
direction (arXiv:2506.09748) -- BEFORE committing a session to a C++
`DinoDenseMatching : IMatchingStage`.

Question this answers: do dense fine-tuned DINOv2 patch features + a plain
mutual-NN correspondence set + RANSAC homography (a) rank the geometrically
correct reference crop above wrong ones by inlier count, and (b) give a
homography whose projected frame-centre lands near ground truth -- on the
exact UAV-VisLoc cross-domain content where ORB/DISK+LightGlue plateau?

If yes -> build the C++ stage. If no -> rethink first. Same "prototype
before building" discipline as the same-domain experiment / batch fusion.

Uses models/dinov2_s_finetuned_dense.onnx (export_retrieval_backbone.py
--dense) and scripts/uavvisloc_grid.py's grid math. Flight 01 only, a
sample of ground-truth frames.

Run from ../CV_IP/cv_env.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uavvisloc_grid as grid  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INPUT_SIZE = 224
GRID = 16                       # 224 / 14 patch size
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
M_PER_DEG_LAT = 111320.0


class DenseEmbedder:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        blob = np.transpose(rgb, (2, 0, 1))[None]           # [1,3,224,224]
        tok = self.sess.run(["patch_tokens"], {"image": blob})[0][0]  # [256,384]
        tok = tok / (np.linalg.norm(tok, axis=1, keepdims=True) + 1e-8)
        return tok.astype(np.float32)


def patch_centres_px(side_px):
    """Pixel centres (in a side_px x side_px image) of the 16x16 patch grid,
    row-major to match token order [row*16 + col]."""
    step = side_px / GRID
    c = (np.arange(GRID) + 0.5) * step
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # [256,2] (x,y)


def match_and_homography(frame_tok, crop_tok, frame_side, crop_side, cos_thr,
                         ratio=1.0):
    """Mutual-NN on token cosine sim + absolute cosine threshold -> RANSAC
    homography. Returns (n_inliers, H) or (0, None).

    NOTE: a cosine *ratio* test (best vs 2nd-best token match) was tried here and
    is useless for this feature space -- DINOv2 patch tokens are locally smooth,
    so adjacent patches have near-identical cosine and best-minus-second is
    always tiny; any ratio margin >0 zeroes out every match. `ratio` kept as a
    no-op arg so the sweep CLI still parses; the absolute threshold is the lever.
    """
    S = frame_tok @ crop_tok.T                      # [256,256] cosine
    f2c = S.argmax(axis=1)
    c2f = S.argmax(axis=0)
    fi = np.arange(S.shape[0])
    mutual = c2f[f2c] == fi
    keep = mutual & (S[fi, f2c] >= cos_thr)
    if keep.sum() < 8:
        return 0, None
    fpx = patch_centres_px(frame_side)[fi[keep]]
    cpx = patch_centres_px(crop_side)[f2c[keep]]
    H, mask = cv2.findHomography(fpx, cpx, cv2.RANSAC, 5.0)
    if H is None:
        return 0, None
    return int(mask.sum()), H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="01")
    ap.add_argument("--model", default="models/dinov2_s_finetuned_dense.onnx")
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--n-decoys", type=int, default=9)
    ap.add_argument("--cos-thr", type=float, default=0.5)
    ap.add_argument("--ratio", type=float, default=1.0, help="cosine ratio test; <1 stricter")
    ap.add_argument("--hard-decoys", action="store_true",
                    help="Use the retrieval model's own top-k grid cells as decoys "
                         "(the real operating condition) instead of random far cells.")
    ap.add_argument("--retrieval-model", default="models/dinov2_s_finetuned_retrieval.onnx")
    args = ap.parse_args()

    emb = DenseEmbedder(REPO / args.model)
    ret = None
    if args.hard_decoys:
        ret = ort.InferenceSession(str(REPO / args.retrieval_model),
                                   providers=["CPUExecutionProvider"])

    def gemb(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        blob = np.transpose(rgb, (2, 0, 1))[None]
        v = ret.run(["embedding"], {"image": blob})[0][0]
        return v / (np.linalg.norm(v) + 1e-8)
    flight = args.flight
    map_img = cv2.imread(str(REPO / "Images" / f"map_clean_uavvisloc_{flight}.png"))
    bounds = grid.load_bounds(f"satellite{flight}.tif")
    lt_lat, lt_lon, rb_lat, rb_lon = bounds
    glats, glngs = grid.generate_grid(*bounds)
    mlng = grid.m_per_deg_lng((lt_lat + rb_lat) / 2.0)

    gt = list(csv.DictReader(open(REPO / "CSV Files" / f"ground_truth_uavvisloc_{flight}.csv")))
    step = max(1, len(gt) // args.n_frames)
    rows = gt[::step][:args.n_frames]

    crop_gembs = None
    if args.hard_decoys:
        ce = []
        for gi in range(len(glats)):
            c = grid.crop_at(map_img, glats[gi], glngs[gi], bounds)
            ce.append(gemb(c) if (c is not None and c.size) else np.zeros(384, np.float32))
        crop_gembs = np.stack(ce)
        print(f"  (hard-decoy mode: {len(ce)} grid cells embedded for retrieval)")

    rng = np.random.default_rng(0)
    true_win, pos_err_true, margins = 0, [], []
    n_eval = 0
    for row in rows:
        lat, lng, label = float(row["Lat"]), float(row["Lng"]), row["Label"]
        frame = cv2.imread(str(REPO / "Datasets" / "UAV_VisLoc_dataset" / flight / "drone" / label))
        if frame is None:
            continue
        d = np.hypot((glats - lat) * M_PER_DEG_LAT, (glngs - lng) * mlng)
        true_i = int(d.argmin())
        if args.hard_decoys:
            fg = gemb(frame)
            order = np.argsort(-(crop_gembs @ fg))
            decoys = np.array([i for i in order if i != true_i][:args.n_decoys])
        else:
            cand_pool = np.where((d > 300) & (d < 2000))[0]
            if len(cand_pool) < args.n_decoys:
                continue
            decoys = rng.choice(cand_pool, args.n_decoys, replace=False)

        ft = emb(frame)
        fs = INPUT_SIZE

        def score(gi):
            crop = grid.crop_at(map_img, glats[gi], glngs[gi], bounds)
            if crop is None or crop.size == 0:
                return 0, None, None
            n, H = match_and_homography(ft, emb(crop), fs, INPUT_SIZE, args.cos_thr, args.ratio)
            return n, H, crop

        n_true, H_true, crop_true = score(true_i)
        decoy_ns = [score(gi)[0] for gi in decoys]
        best_decoy = max(decoy_ns) if decoy_ns else 0
        n_eval += 1
        margins.append(n_true - best_decoy)
        if n_true > best_decoy:
            true_win += 1

        if H_true is not None and crop_true is not None:
            # project frame centre -> crop px -> lat/lng, compare to GT
            fc = np.array([[[fs / 2.0, fs / 2.0]]], dtype=np.float32)
            cp = cv2.perspectiveTransform(fc, H_true)[0, 0]   # (x,y) in 224-crop space
            ch, cw = crop_true.shape[:2]
            # crop was 300 m wide centred on grid cell; map 224-space -> metre offset
            mx = (cp[0] / fs - 0.5) * grid.CROP_SIZE_M
            my = (cp[1] / fs - 0.5) * grid.CROP_SIZE_M
            est_lat = glats[true_i] - my / M_PER_DEG_LAT
            est_lng = glngs[true_i] + mx / mlng
            err = np.hypot((est_lat - lat) * M_PER_DEG_LAT, (est_lng - lng) * mlng)
            pos_err_true.append(err)

    print(f"\nflight {flight}  n={n_eval} frames  cos_thr={args.cos_thr}  decoys={args.n_decoys}")
    print(f"  true crop beats all decoys on inliers: {true_win}/{n_eval} "
          f"({100*true_win/max(1,n_eval):.0f}%)")
    m = np.array(margins)
    print(f"  inlier margin (true - best decoy): median {np.median(m):+.0f}  "
          f"mean {m.mean():+.1f}  (>0 on {(m>0).mean():.0%})")
    if pos_err_true:
        e = np.array(pos_err_true)
        print(f"  position error via H on true crop: median {np.median(e):.0f} m  "
              f"mean {e.mean():.0f} m  <50m {(e<50).mean():.0%}  <100m {(e<100).mean():.0%}")


if __name__ == "__main__":
    main()
