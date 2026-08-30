#!/usr/bin/env python3
"""
STRATEGY.md Phase 2+ Option A2, Stage A2.0 -- the make-or-break de-risk check,
BEFORE any training, ONNX export, or C++.

Question: does a neighbourhood-consensus 4D-conv module (NCNet, MIT, vendored
in scripts/vendor/ncnet/) on top of our fine-tuned dense DINOv2 patch features
raise HARD-DECOY crop discrimination -- the metric stuck at ~28-33% across every
plain-matching variant tried in Option A (see INVESTIGATION_LOG.md)?

  dense DINOv2 tokens (frame, crop), each [16,16,384], L2-normed per token
    -> 4D correlation tensor [1,1,16,16,16,16]
    -> MutualMatching -> NeighConsensus (3x Conv4d+ReLU, symmetric) -> MutualMatching
    -> per-cell argmax + mutual-NN + score threshold -> RANSAC homography
    -> inlier count (the discriminator) + projected-centre pose error

Compares --mode plain (no consensus, today's baseline) vs --mode ncnet.
--ncnet-weights loads a checkpoint's NeighConsensus.* params; omitted = random
init (tests the mechanism/architecture itself first, per the de-risk plan).

GATE: ncnet hard-decoy discrimination must rise meaningfully (target >50%, from
~30%). If it stays ~30% with both random AND pretrained weights -> Option A2 is
dead, stop and consolidate.

Run from ../CV_IP/cv_env.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "vendor" / "ncnet"))
import uavvisloc_grid as grid  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
M_PER_DEG_LAT = 111320.0
INPUT_SIZE = 224
G = 16  # 224 / 14 patch grid


def load_dino(weights, img_size=224):
    import torch, timm
    m = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0,
                          img_size=224, dynamic_img_size=True)
    m.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    m.eval()
    return m


def dino_grid(m, bgr):
    """-> torch tensor [1, 384, G, G], L2-normed per spatial token (channel dim)."""
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    blob = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None])
    with torch.no_grad():
        f = m.forward_features(blob)                        # [1, 1+G*G, 384]
    tok = f[0, m.num_prefix_tokens:, :]                     # [G*G, 384]
    tok = tok / (tok.norm(dim=1, keepdim=True) + 1e-8)
    return tok.t().reshape(1, 384, G, G)                    # [1, 384, G, G]


def build_ncnet(weights=None, kernel_sizes=(3, 3, 3), channels=(10, 10, 1)):
    import torch
    from ncnet_consensus import NeighConsensus
    nc = NeighConsensus(use_cuda=False, kernel_sizes=list(kernel_sizes),
                        channels=list(channels), symmetric_mode=True)
    nc.eval()
    if weights:
        ckpt = torch.load(weights, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        nc_sd = {k.replace("NeighConsensus.", ""): v for k, v in sd.items()
                 if k.startswith("NeighConsensus.")}
        missing = nc.load_state_dict(nc_sd, strict=False)
        print(f"  loaded NeighConsensus weights ({len(nc_sd)} tensors; {missing})")
    return nc


def patch_centres(side):
    step = side / G
    c = (np.arange(G) + 0.5) * step
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float32)  # [G*G, 2] (x,y), row-major


def corr_from_grids(fa, fb):
    """NCNet FeatureCorrelation 4D path (normalization=False -- tokens already
    L2-normed). -> [1,1,G,G,G,G] indexed [b,1,iA,jA,iB,jB]."""
    import torch
    b, c, h, w = fa.shape
    A = fa.view(b, c, h * w).transpose(1, 2)   # [b, hw, c]
    B = fb.view(b, c, h * w)                   # [b, c, hw]
    return torch.bmm(A, B).view(b, h, w, h, w).unsqueeze(1)


def match_ncnet(fa, fb, nc, use_consensus, frame_side, crop_side, score_thr):
    import torch
    from ncnet_consensus import MutualMatching
    with torch.no_grad():
        corr = corr_from_grids(fa, fb)            # [1,1,G,G,G,G]
        corr = MutualMatching(corr)
        if use_consensus:
            corr = nc(corr)
            corr = MutualMatching(corr)
        c = corr[0, 0].reshape(G * G, G * G).numpy()   # [A, B]
    a2b = c.argmax(1)
    b2a = c.argmax(0)
    ai = np.arange(G * G)
    mutual = b2a[a2b] == ai
    sc = c[ai, a2b]
    sc = sc / (sc.max() + 1e-9)                        # normalize scores to [0,1] for a stable thr
    keep = mutual & (sc >= score_thr)
    if keep.sum() < 8:
        return 0, None
    fpx = patch_centres(frame_side)[ai[keep]]
    cpx = patch_centres(crop_side)[a2b[keep]]
    H, mask = cv2.findHomography(fpx, cpx, cv2.RANSAC, frame_side / G)
    if H is None:
        return 0, None
    return int(mask.sum()), H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="01")
    ap.add_argument("--dino-weights", default="checkpoints/dinov2_s_finetuned.pt")
    ap.add_argument("--mode", choices=["plain", "ncnet", "both"], default="both")
    ap.add_argument("--ncnet-weights", default=None)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--n-decoys", type=int, default=9)
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dino = load_dino(str(REPO / args.dino_weights))
    nc = build_ncnet(str(REPO / args.ncnet_weights) if args.ncnet_weights else None)

    # cheap global embedder (mean-pooled dense grid) for the hard-decoy retrieval
    def gemb_grid(fg):
        v = fg.mean(dim=(2, 3))[0].numpy()
        return v / (np.linalg.norm(v) + 1e-8)

    flight = args.flight
    map_img = cv2.imread(str(REPO / "Images" / f"map_clean_uavvisloc_{flight}.png"))
    bounds = grid.load_bounds(f"satellite{flight}.tif")
    glats, glngs = grid.generate_grid(*bounds)
    mlng = grid.m_per_deg_lng((bounds[0] + bounds[2]) / 2.0)

    gt = list(csv.DictReader(open(REPO / "CSV Files" / f"ground_truth_uavvisloc_{flight}.csv")))
    rows = gt[:: max(1, len(gt) // args.n_frames)][:args.n_frames]

    # embed all grid cells once (dense grid tensors are heavy; cache)
    cell_grids = {}
    cell_gemb = np.zeros((len(glats), 384), np.float32)
    for gi in range(len(glats)):
        c = grid.crop_at(map_img, glats[gi], glngs[gi], bounds)
        if c is None or c.size == 0:
            continue
        fg = dino_grid(dino, c)
        cell_grids[gi] = fg
        cell_gemb[gi] = gemb_grid(fg)
    print(f"flight {flight}: {len(cell_grids)} grid cells embedded")

    modes = ["plain", "ncnet"] if args.mode == "both" else [args.mode]
    stats = {mmode: {"win": 0, "n": 0, "perr": []} for mmode in modes}

    for row in rows:
        lat, lng, label = float(row["Lat"]), float(row["Lng"]), row["Label"]
        fr = cv2.imread(str(REPO / "Datasets" / "UAV_VisLoc_dataset" / flight / "drone" / label))
        if fr is None:
            continue
        d = np.hypot((glats - lat) * M_PER_DEG_LAT, (glngs - lng) * mlng)
        ti = int(d.argmin())
        if ti not in cell_grids:
            continue
        fg = dino_grid(dino, fr)
        order = np.argsort(-(cell_gemb @ gemb_grid(fg)))
        decoys = [i for i in order if i != ti and i in cell_grids][:args.n_decoys]
        if len(decoys) < args.n_decoys:
            continue

        for mmode in modes:
            use_c = (mmode == "ncnet")
            n_t, H_t = match_ncnet(fg, cell_grids[ti], nc, use_c, INPUT_SIZE, INPUT_SIZE, args.score_thr)
            n_d = [match_ncnet(fg, cell_grids[gi], nc, use_c, INPUT_SIZE, INPUT_SIZE, args.score_thr)[0]
                   for gi in decoys]
            st = stats[mmode]
            st["n"] += 1
            if n_t > max(n_d + [0]):
                st["win"] += 1
            if H_t is not None:
                cp = cv2.perspectiveTransform(
                    np.array([[[INPUT_SIZE / 2.0, INPUT_SIZE / 2.0]]], np.float32), H_t)[0, 0]
                mx = (cp[0] / INPUT_SIZE - 0.5) * grid.CROP_SIZE_M
                my = (cp[1] / INPUT_SIZE - 0.5) * grid.CROP_SIZE_M
                elat = glats[ti] - my / M_PER_DEG_LAT
                elng = glngs[ti] + mx / mlng
                st["perr"].append(np.hypot((elat - lat) * M_PER_DEG_LAT, (elng - lng) * mlng))

    print(f"\nflight {flight}  n_frames_eval={stats[modes[0]]['n']}  "
          f"score_thr={args.score_thr}  ncnet_weights={args.ncnet_weights or 'RANDOM INIT'}")
    for mmode in modes:
        st = stats[mmode]
        disc = 100 * st["win"] / max(1, st["n"])
        pe = np.array(st["perr"]) if st["perr"] else np.array([np.nan])
        print(f"  [{mmode:5s}] hard-decoy discrimination {st['win']}/{st['n']} ({disc:.0f}%)   "
              f"pose-on-true-crop median {np.nanmedian(pe):.0f}m  <100m {np.mean(pe < 100):.0%} "
              f"(n={len(st['perr'])})")


if __name__ == "__main__":
    main()
