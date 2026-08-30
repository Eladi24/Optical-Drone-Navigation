#!/usr/bin/env python3
"""
STRATEGY.md Phase 2+ Option A2, Stage A2.0 -- the real de-risk: does TRAINING the
neighbourhood-consensus module (not random init) on our data move hard-decoy crop
discrimination off the ~30% floor?

A deliberately quick, local, CPU-only proof -- NOT the real training pipeline
(that's Stage A2.1/A2.2 on Colab). Trains ONLY the ~23K-param NeighConsensus
module (DINOv2 backbone frozen), NCNet's own weakly-supervised loss (push the
mean-matching-score up for the correct frame/crop pair, down for hard-negative
crops), on flight-01 dev frames, then measures plain vs trained-consensus
discrimination on HELD-OUT flight-01 frames.

GATE: trained-consensus hard-decoy discrimination rises meaningfully (target
>50%, from ~30%). If it stays ~30% even after training -> Option A2 is very
likely dead; stop and consolidate. If it clearly moves -> greenlight A2.1+.

Run from ../CV_IP/cv_env. ~10-15 min on CPU.
"""
import argparse
import csv
import sys
import time
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
G = 16


def load_dino(weights):
    import torch, timm
    m = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0,
                          img_size=224, dynamic_img_size=True)
    m.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def dino_grid(m, bgr):
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    blob = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None])
    with torch.no_grad():
        f = m.forward_features(blob)
    tok = f[0, m.num_prefix_tokens:, :]
    tok = tok / (tok.norm(dim=1, keepdim=True) + 1e-8)
    return tok.t().reshape(1, 384, G, G).contiguous()


def corr(fa, fb):
    import torch
    b, c, h, w = fa.shape
    A = fa.view(b, c, h * w).transpose(1, 2)
    B = fb.view(b, c, h * w)
    return torch.bmm(A, B).view(b, h, w, h, w).unsqueeze(1)


def mean_match_score(corr4d):
    """NCNet weakly-supervised score: MutualMatching then mean of per-A-cell max."""
    from ncnet_consensus import MutualMatching
    c = MutualMatching(corr4d)
    b = c.shape[0]
    cA = c.view(b, G * G, G * G).max(dim=2)[0].mean(dim=1)
    cB = c.view(b, G * G, G * G).max(dim=1)[0].mean(dim=1)
    return (cA + cB) / 2.0


def patch_centres(side):
    step = side / G
    cc = (np.arange(G) + 0.5) * step
    yy, xx = np.meshgrid(cc, cc, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float32)


def discriminate(fg, cell_grids, ti, decoys, nc, use_c, score_thr):
    import torch
    from ncnet_consensus import MutualMatching

    def inl(fb):
        with torch.no_grad():
            c = MutualMatching(corr(fg, fb))
            if use_c:
                c = MutualMatching(nc(c))
            m = c[0, 0].reshape(G * G, G * G).numpy()
        a2b = m.argmax(1); b2a = m.argmax(0); ai = np.arange(G * G)
        sc = m[ai, a2b]; sc = sc / (sc.max() + 1e-9)
        keep = (b2a[a2b] == ai) & (sc >= score_thr)
        if keep.sum() < 8:
            return 0
        H, mask = cv2.findHomography(patch_centres(INPUT_SIZE)[ai[keep]],
                                    patch_centres(INPUT_SIZE)[a2b[keep]],
                                    cv2.RANSAC, INPUT_SIZE / G)
        return 0 if H is None else int(mask.sum())

    n_t = inl(cell_grids[ti])
    n_d = max([inl(cell_grids[gi]) for gi in decoys] + [0])
    return n_t > n_d


def build_samples(dino, flight, cell_grids, cell_gemb, glats, glngs, mlng, t0):
    gt = list(csv.DictReader(open(REPO / "CSV Files" / f"ground_truth_uavvisloc_{flight}.csv")))
    out = []
    for row in gt[::3]:
        lat, lng, label = float(row["Lat"]), float(row["Lng"]), row["Label"]
        fr = cv2.imread(str(REPO / "Datasets" / "UAV_VisLoc_dataset" / flight / "drone" / label))
        if fr is None:
            continue
        d = np.hypot((glats - lat) * M_PER_DEG_LAT, (glngs - lng) * mlng)
        ti = int(d.argmin())
        if ti not in cell_grids:
            continue
        fg = dino_grid(dino, fr)
        fv = fg.mean(dim=(2, 3))[0].numpy()
        fv = fv / (np.linalg.norm(fv) + 1e-8)
        order = np.argsort(-(cell_gemb @ fv))
        decoys = [i for i in order if i != ti and i in cell_grids][:9]
        if len(decoys) < 9:
            continue
        out.append((fg, ti, decoys))
    print(f"  flight {flight}: {len(out)} usable frames ({time.time()-t0:.0f}s)")
    return out


def embed_cells(dino, map_img, glats, glngs, bounds, cell_grids, cell_gemb, t0):
    for gi in range(len(glats)):
        c = grid.crop_at(map_img, glats[gi], glngs[gi], bounds)
        if c is None or c.size == 0:
            continue
        fgd = dino_grid(dino, c)
        cell_grids[gi] = fgd
        v = fgd.mean(dim=(2, 3))[0].numpy()
        cell_gemb[gi] = v / (np.linalg.norm(v) + 1e-8)


def main():
    import torch
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="01")
    ap.add_argument("--dino-weights", default="checkpoints/dinov2_s_finetuned.pt")
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--channels", default="6,6,1")
    ap.add_argument("--symmetric", type=int, default=0)   # 0 = off (half the memory) for the de-risk
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dino = load_dino(str(REPO / args.dino_weights))
    from ncnet_consensus import NeighConsensus
    ch = [int(x) for x in args.channels.split(",")]
    nc = NeighConsensus(use_cuda=False, kernel_sizes=[3] * len(ch), channels=ch,
                        symmetric_mode=bool(args.symmetric))
    n_params = sum(p.numel() for p in nc.parameters())
    print(f"NeighConsensus: channels={ch} symmetric={bool(args.symmetric)}  {n_params} params")

    t0 = time.time()
    fl = args.flight
    map_img = cv2.imread(str(REPO / "Images" / f"map_clean_uavvisloc_{fl}.png"))
    bounds = grid.load_bounds(f"satellite{fl}.tif")
    glats, glngs = grid.generate_grid(*bounds)
    mlng = grid.m_per_deg_lng((bounds[0] + bounds[2]) / 2.0)
    cg, ge = {}, np.zeros((len(glats), 384), np.float32)
    embed_cells(dino, map_img, glats, glngs, bounds, cg, ge, t0)
    print(f"  flight {fl}: {len(cg)} cells embedded ({time.time()-t0:.0f}s)")

    samples = build_samples(dino, fl, cg, ge, glats, glngs, mlng, t0)
    # disjoint train/eval by TRUE grid cell so eval crops are genuinely unseen
    rng.shuffle(samples)
    eval_s = samples[:args.n_eval]
    eval_ti = {ti for _, ti, _ in eval_s}
    train_s = [s for s in samples[args.n_eval:] if s[1] not in eval_ti]
    print(f"{len(train_s)} train / {len(eval_s)} eval  ({time.time()-t0:.0f}s)")

    def eval_disc(use_c):
        nc.eval()
        with torch.no_grad():
            v = np.mean([discriminate(fg, cg, ti, dc, nc, use_c, args.score_thr)
                         for fg, ti, dc in eval_s])
        nc.train()
        return v

    base_plain = eval_disc(False)
    print(f"baseline  plain={base_plain:.0%}   random-consensus={eval_disc(True):.0%}")

    opt = torch.optim.Adam(nc.parameters(), lr=args.lr)
    best = 0.0
    for step in range(args.steps):
        fg, ti, decoys = train_s[rng.integers(len(train_s))]
        negs = rng.choice(decoys, size=min(args.n_neg, len(decoys)), replace=False)
        opt.zero_grad()
        # ONE autograd graph alive at a time -- backward per term, accumulate grads.
        s_pos = mean_match_score(nc(corr(fg, cg[ti]))).squeeze()
        (-s_pos).backward()
        sp = s_pos.item()
        sn = 0.0
        for gi in negs:
            s_neg = mean_match_score(nc(corr(fg, cg[int(gi)]))).squeeze()
            (s_neg / len(negs)).backward()
            sn += s_neg.item() / len(negs)
        torch.nn.utils.clip_grad_norm_(nc.parameters(), 1.0)
        opt.step()
        if (step + 1) % 20 == 0:
            ev = eval_disc(True)
            best = max(best, ev)
            print(f"  step {step+1:4d}  s_pos {sp:.3f}  s_neg {sn:.3f}  "
                  f"eval_disc {ev:.0%} (best {best:.0%})  ({time.time()-t0:.0f}s)")

    print(f"\nflight {fl}  eval n={len(eval_s)}  score_thr={args.score_thr}")
    print(f"  plain (no consensus)          : {base_plain:.0%}")
    print(f"  trained NeighConsensus (best) : {best:.0%}")
    print(f"  --> {'GREENLIGHT A2.1+' if best >= base_plain + 0.15 else 'weak/negative -- likely stop'}")


if __name__ == "__main__":
    main()
