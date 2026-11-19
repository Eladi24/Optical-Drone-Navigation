#!/usr/bin/env python3
"""
Prototype: AnyLoc-style VLAD aggregation over DINOv2-S dense patch features as
the retrieval descriptor, measured with the exact same offline recall@{1,5,10}
protocol as scripts/eval_retrieval_recall.py so the numbers drop straight into
the RESULTS.md / CLAUDE.md retrieval tables.

Why this exists (STRATEGY.md / RESULTS.md route #2): every learned-retrieval
result so far (split_deit, split_finetuned_disk, ...) used a single POOLED
embedding -- mean-pool or CLS token. AnyLoc (arXiv:2308.00688) showed that
VLAD-aggregating the *dense per-patch* features of a frozen DINOv2 backbone is
substantially stronger for cross-view place recognition (NetVLAD 31.1% ->
AnyLoc-class methods 56-62% recall@1 on the AnyVisLoc benchmark). This project
never built that aggregation. This script tests whether it moves this project's
own recall@k on UAV-VisLoc, in Python, before any C++ (OnnxVladRetrieval) work.

VLAD recipe (AnyLoc's default, minus optional PCA-whitening):
  1. dense patch descriptors from DINOv2-S  ([n_patch, D], D=384)
  2. hard-assign each patch to the nearest of k vocabulary centroids
  3. per centroid: sum the residuals (patch - centroid) of its assigned patches
  4. intra-normalize (L2 per centroid block), concatenate  -> [k*D]
  5. optional signed square-root power-normalization, then global L2

Vocabulary is k-means over patch descriptors from the DEV flights only (01, 03)
-- training data, so held-out flights stay a genuine zero-shot check. Cached to
models/vlad_vocab_<tag>_k<k>.npy.

Usage:
    source ../CV_IP/cv_env/bin/activate

    # Frozen ImageNet DINOv2-S (live PyTorch), k=32 -- the headline comparison
    python3 scripts/proto_vlad_retrieval.py --weights none --clusters 32

    # Fine-tuned dense checkpoint (already exported)
    python3 scripts/proto_vlad_retrieval.py \
        --weights models/dinov2_s_finetuned_dense.onnx --clusters 32
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import uavvisloc_grid as grid

REPO = Path(__file__).parent.parent
CSV_DIR = REPO / "CSV Files"
IMAGES_DIR = REPO / "Images"
DATASET_DIR = REPO / "Datasets" / "UAV_VisLoc_dataset"
MODELS_DIR = REPO / "models"

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEV_FLIGHTS = ["01", "03"]
VALIDATION_FLIGHTS = ["04", "08", "10"]
HELD_OUT_FLIGHTS = ["02", "05", "06", "11"]
ALL_FLIGHTS = DEV_FLIGHTS + VALIDATION_FLIGHTS + HELD_OUT_FLIGHTS


class DenseEmbedder:
    """dense patch descriptors [n_patch, D], L2-normalized per patch.
    --weights none -> live PyTorch frozen DINOv2-S (matches
    models/dinov2_s_retrieval.onnx's weights); a .onnx path -> that dense
    export (must output 'patch_tokens', i.e. built with
    export_retrieval_backbone.py --dense)."""

    def __init__(self, weights):
        self.use_onnx = weights not in (None, "none")
        if self.use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
            self.out_name = self.session.get_outputs()[0].name
            print(f"DenseEmbedder: ONNX Runtime, model={weights}, output={self.out_name}")
        else:
            import timm
            import torch
            self.torch = torch
            base = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                     num_classes=0, img_size=INPUT_SIZE)
            base.eval()
            self.n_prefix = int(getattr(base, "num_prefix_tokens", 1))
            self.model = base
            print("DenseEmbedder: live PyTorch, frozen ImageNet DINOv2-S (no fine-tune)")

    def _preprocess(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

    def embed_dense(self, bgr_image):
        x = self._preprocess(bgr_image)
        if self.use_onnx:
            out = self.session.run([self.out_name], {"image": x})[0][0]   # [n_patch, D]
        else:
            with self.torch.no_grad():
                feats = self.model.forward_features(self.torch.from_numpy(x))
            out = feats[0, self.n_prefix:, :].numpy()
        out = out.astype(np.float32)
        out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return out


def vlad_encode(patch_descs, centroids, power_norm=True):
    """patch_descs [P, D], centroids [k, D] -> VLAD vector [k*D], L2-normalized."""
    k, D = centroids.shape
    # nearest centroid per patch
    d2 = ((patch_descs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)   # [P, k]
    assign = np.argmin(d2, axis=1)                                             # [P]
    V = np.zeros((k, D), dtype=np.float32)
    for c in range(k):
        m = assign == c
        if np.any(m):
            V[c] = (patch_descs[m] - centroids[c]).sum(axis=0)
    # intra-normalization (per centroid block)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    v = V.ravel()
    if power_norm:
        v = np.sign(v) * np.sqrt(np.abs(v))
    v /= (np.linalg.norm(v) + 1e-9)
    return v


def iter_crops(flight, map_img, bounds):
    grid_lats, grid_lngs = grid.generate_grid(*bounds)
    for i in range(len(grid_lats)):
        crop = grid.crop_at(map_img, grid_lats[i], grid_lngs[i], bounds)
        if crop is None or crop.size == 0:
            continue
        yield i, grid_lats[i], grid_lngs[i], crop


def build_vocabulary(embedder, weights_tag, n_clusters, crops_per_flight):
    cache = MODELS_DIR / f"vlad_vocab_{weights_tag}_k{n_clusters}.npy"
    if cache.exists():
        print(f"vocab: loading cached {cache.name}")
        return np.load(cache)

    print(f"vocab: k-means k={n_clusters} over dev flights {DEV_FLIGHTS} "
          f"(<= {crops_per_flight} crops/flight)")
    rng = np.random.default_rng(0)
    descs = []
    for flight in DEV_FLIGHTS:
        map_img = cv2.imread(str(IMAGES_DIR / f"map_clean_uavvisloc_{flight}.png"))
        bounds = grid.load_bounds(f"satellite{flight}.tif")
        all_crops = list(iter_crops(flight, map_img, bounds))
        pick = rng.choice(len(all_crops), size=min(crops_per_flight, len(all_crops)),
                          replace=False)
        for j in pick:
            descs.append(embedder.embed_dense(all_crops[j][3]))
        print(f"  flight {flight}: sampled {len(pick)}/{len(all_crops)} crops")
    descs = np.concatenate(descs, axis=0)
    print(f"  {descs.shape[0]} patch descriptors -> fitting k-means")

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=4096,
                         n_init=3, max_iter=200)
    km.fit(descs)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    np.save(cache, centroids)
    print(f"vocab: saved {cache.name}  ({centroids.nbytes/1e6:.1f} MB)")
    return centroids


def eval_flight(flight, embedder, centroids, power_norm, queries=("full",), frame_stride=1):
    map_path = IMAGES_DIR / f"map_clean_uavvisloc_{flight}.png"
    gt_path = CSV_DIR / f"ground_truth_uavvisloc_{flight}.csv"
    map_img = cv2.imread(str(map_path))
    if map_img is None:
        print(f"  SKIP flight {flight}: no {map_path.name}")
        return None
    bounds = grid.load_bounds(f"satellite{flight}.tif")

    crop_vecs, lats, lngs = [], [], []
    for i, clat, clng, crop in iter_crops(flight, map_img, bounds):
        crop_vecs.append(vlad_encode(embedder.embed_dense(crop), centroids, power_norm))
        lats.append(clat); lngs.append(clng)
    crop_vecs = np.stack(crop_vecs)
    lats, lngs = np.array(lats), np.array(lngs)
    print(f"  flight {flight}: {len(crop_vecs)} grid cells encoded (VLAD dim {crop_vecs.shape[1]})")

    mpp = grid.map_mpp(map_img, bounds)
    alt = grid.load_altitude_cache(flight) if any(q != "full" for q in queries) else None
    headings = grid.load_headings(flight) if "northup" in queries else {}
    ranks = {q: [] for q in queries}
    for row in csv.DictReader(open(gt_path, newline="")):
        if int(row["Frame"]) % frame_stride:
            continue
        lat, lng, label = float(row["Lat"]), float(row["Lng"]), row["Label"]
        frame = cv2.imread(str(DATASET_DIR / flight / "drone" / label))
        if frame is None:
            continue
        true_idx = grid.nearest_grid_index(lat, lng, lats, lngs)
        gsd = grid.frame_gsd(alt, frame.shape[1]) if alt else None
        for q in queries:
            query = grid.make_query(frame, q, gsd, mpp, headings.get(int(row["Frame"])))
            fvec = vlad_encode(embedder.embed_dense(query), centroids, power_norm)
            order = np.argsort(-(crop_vecs @ fvec))
            ranks[q].append(int(np.where(order == true_idx)[0][0]) + 1)

    for q in queries:
        ranks[q] = np.array(ranks[q])
        r = {k: float((ranks[q] <= k).mean()) for k in (1, 5, 10)}
        print(f"    [{q:>7}] n={len(ranks[q])}  recall@1={r[1]:.1%}  recall@5={r[5]:.1%}  recall@10={r[10]:.1%}")
    return ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="none")
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--vocab-crops-per-flight", type=int, default=150)
    ap.add_argument("--no-power-norm", action="store_true")
    ap.add_argument("--flights", default=None)
    ap.add_argument("--queries", default="full",
                    help="Comma-separated query modes: full (legacy whole photo), crop (C++ GSD-normalised "
                         "centre crop), northup (crop + -heading pre-warp). See uavvisloc_grid.make_query().")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="score every Nth GT frame (consecutive stills overlap heavily)")
    args = ap.parse_args()

    flights = args.flights.split(",") if args.flights else ALL_FLIGHTS
    tag = "frozen" if args.weights in (None, "none") else \
        hashlib.md5(args.weights.encode()).hexdigest()[:8]
    embedder = DenseEmbedder(args.weights)
    centroids = build_vocabulary(embedder, tag, args.clusters, args.vocab_crops_per_flight)
    power_norm = not args.no_power_norm

    queries = tuple(args.queries.split(","))
    per_flight = {}
    for flight in flights:
        print(f"Evaluating flight {flight}...")
        ranks = eval_flight(flight, embedder, centroids, power_norm, queries, args.frame_stride)
        if ranks:
            per_flight[flight] = ranks
    from eval_retrieval_recall import report_pooled
    report_pooled(per_flight, queries,
                  f"VLAD (k={args.clusters}, power_norm={power_norm}, weights={args.weights})", args.json_out)


if __name__ == "__main__":
    main()
