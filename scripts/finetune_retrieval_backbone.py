#!/usr/bin/env python3
"""
Stage 1b of the DINOv2-S retrieval fine-tuning pipeline (see
.claude/plans/enumerated-yawning-nygaard.md). Reads
scripts/prepare_finetune_dataset.py's manifest, fine-tunes DINOv2-S (via
timm, matching scripts/export_retrieval_backbone.py's architecture exactly)
with a temperature-scaled InfoNCE loss against mined hard negatives + easy
negatives + free in-batch negatives.

Only the last N_UNFROZEN_BLOCKS transformer blocks + final norm are
unfrozen -- a deliberate, parameter-efficient choice given this project's own
documented overfitting risk on a small dataset (a few thousand frames for a
21M-param ViT), not the default. Checkpoints are saved as a bare
`model.state_dict()` .pt file -- directly loadable by
export_retrieval_backbone.py's `--weights` argument, same architecture, no
key remapping needed.

Local GPU (NVIDIA GeForce MX330, 2GB, WSL2 passthrough) was tried and
confirmed non-functional for real CUDA compute this session (detection
succeeds, every actual GPU operation fails with "CUDA error: operation not
supported" -- a driver/WSL2-passthrough limitation, not fixable from inside
this environment) -- see CLAUDE.md's Investigation Log. This script is
device-agnostic (--device auto/cuda/cpu) so the exact same code runs
unchanged in a Colab notebook (the actual training venue) as it does here for
the CPU-only --smoke-test.

Usage:
    source ../CV_IP/cv_env/bin/activate

    # CPU smoke test -- tiny subset, few epochs, confirms the loop runs end to
    # end and produces a loadable checkpoint. Not a real training run.
    python3 scripts/finetune_retrieval_backbone.py --smoke-test

    # Real run (in a Colab notebook, or any machine with working CUDA):
    python3 scripts/finetune_retrieval_backbone.py --device cuda \\
        --epochs 15 --batch-size 16 --checkpoint checkpoints/dinov2_s_finetuned.pt
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
import uavvisloc_grid as grid

REPO = Path(__file__).parent.parent
IMAGES_DIR = REPO / "Images"
MANIFEST_PATH = REPO / "CSV Files" / "finetune_manifest.csv"

INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_UNFROZEN_BLOCKS = 2       # only the last 2 transformer blocks + final norm are trainable
TEMPERATURE = 0.07          # standard InfoNCE default (CLIP/SimCLR-family)
N_NEGATIVES_PER_SAMPLE = 6  # matches prepare_finetune_dataset.py's N_HARD+N_EASY defaults


def build_model(freeze_all_but_last_n: int = N_UNFROZEN_BLOCKS):
    import timm
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                               num_classes=0, img_size=INPUT_SIZE)
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks[-freeze_all_but_last_n:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.norm.parameters():
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: vit_small_patch14_dinov2 -- {n_trainable:,}/{n_total:,} params trainable "
          f"(last {freeze_all_but_last_n} blocks + norm)")
    return model


class FinetunePairDataset(Dataset):
    """Reads prepare_finetune_dataset.py's manifest; crops the satellite side
    on the fly from the cached map PNG (uavvisloc_grid.crop_at), loads the
    drone JPEG directly. Pads/duplicates negatives to a fixed count per
    sample (N_NEGATIVES_PER_SAMPLE) so batches stack without a custom
    collate_fn -- a sample that mined fewer negatives than the target count
    (rare, only near a flight's map edges) reuses its own negatives rather
    than leaving a ragged tensor."""

    def __init__(self, manifest_path, subset_n=None):
        rows = list(csv.DictReader(open(manifest_path, newline="")))
        if subset_n is not None:
            rows = rows[:subset_n]
        self.rows = rows
        self._map_cache = {}   # flight -> (cv2 image, bounds)
        self.drone_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomRotation(180),  # drone heading is arbitrary, unlike the map crop
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.crop_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.rows)

    def _get_map(self, flight):
        if flight not in self._map_cache:
            import cv2
            img = cv2.imread(str(IMAGES_DIR / f"map_clean_uavvisloc_{flight}.png"))
            bounds = grid.load_bounds(f"satellite{flight}.tif")
            self._map_cache[flight] = (img, bounds)
        return self._map_cache[flight]

    def _crop(self, flight, lat, lng):
        import cv2
        map_img, bounds = self._get_map(flight)
        crop = grid.crop_at(map_img, lat, lng, bounds)
        if crop is None or crop.size == 0:
            crop = np.zeros((64, 64, 3), dtype=np.uint8)  # degenerate edge case, rare
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        import cv2
        row = self.rows[idx]
        flight = row["flight"]

        drone_bgr = cv2.imread(str(REPO / row["drone_image"]))
        drone_rgb = cv2.cvtColor(drone_bgr, cv2.COLOR_BGR2RGB)
        anchor = self.drone_tf(drone_rgb)

        pos_crop = self._crop(flight, float(row["pos_lat"]), float(row["pos_lng"]))
        positive = self.crop_tf(pos_crop)

        neg_coords = []
        for part in (row["hard_negs"], row["easy_negs"]):
            for pair in part.split(";"):
                if pair:
                    lat, lng = pair.split(",")
                    neg_coords.append((float(lat), float(lng)))
        if not neg_coords:
            neg_coords = [(float(row["pos_lat"]), float(row["pos_lng"]))]  # last-resort fallback
        while len(neg_coords) < N_NEGATIVES_PER_SAMPLE:
            neg_coords.append(neg_coords[len(neg_coords) % len(neg_coords)])
        neg_coords = neg_coords[:N_NEGATIVES_PER_SAMPLE]
        negatives = torch.stack([self.crop_tf(self._crop(flight, lat, lng))
                                  for lat, lng in neg_coords])

        return anchor, positive, negatives


def info_nce_loss(anchor_emb, pos_emb, neg_emb, temperature=TEMPERATURE):
    """anchor_emb, pos_emb: (B, D) L2-normalized. neg_emb: (B, K, D) L2-normalized,
    per-sample own negatives. In-batch other-samples' positives are free extra
    negatives (standard practice) via the full anchor-vs-all-positives matrix."""
    B = anchor_emb.shape[0]
    sim_pos_matrix = anchor_emb @ pos_emb.T                       # (B, B), diagonal = true positive
    sim_own_neg = torch.einsum("bd,bkd->bk", anchor_emb, neg_emb)  # (B, K)
    logits = torch.cat([sim_pos_matrix, sim_own_neg], dim=1) / temperature
    targets = torch.arange(B, device=anchor_emb.device)
    return F.cross_entropy(logits, targets)


def train(args):
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                           else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")

    subset_n = 20 if args.smoke_test else None
    dataset = FinetunePairDataset(MANIFEST_PATH, subset_n=subset_n)
    print(f"Dataset: {len(dataset)} training pairs"
          + (" (SMOKE TEST subset)" if args.smoke_test else ""))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, drop_last=True)

    model = build_model(args.unfrozen_blocks).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    epochs = 1 if args.smoke_test else args.epochs
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for anchor, positive, negatives in loader:
            anchor, positive, negatives = anchor.to(device), positive.to(device), negatives.to(device)
            B, K = negatives.shape[0], negatives.shape[1]

            anchor_emb = F.normalize(model(anchor), dim=-1)
            pos_emb = F.normalize(model(positive), dim=-1)
            neg_flat = negatives.view(B * K, *negatives.shape[2:])
            neg_emb = F.normalize(model(neg_flat), dim=-1).view(B, K, -1)

            loss = info_nce_loss(anchor_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if args.smoke_test and n_batches >= 2:
                break

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}: avg InfoNCE loss = {avg_loss:.4f} ({n_batches} batches)")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-test", action="store_true",
                     help="Tiny subset (20 pairs), 1 epoch, 2 batches -- confirms the loop runs "
                          "end to end. Not a real training run.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--unfrozen-blocks", type=int, default=N_UNFROZEN_BLOCKS)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dinov2_s_finetuned.pt"))
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found at {MANIFEST_PATH} -- run scripts/prepare_finetune_dataset.py first.")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
