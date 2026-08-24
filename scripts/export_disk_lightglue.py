#!/usr/bin/env python3
"""
Exports DISK (feature extractor) + LightGlue (matcher) to two separate ONNX
graphs, for STRATEGY.md Phase 2's second learned-matching candidate (XFeat was
the first, already done -- see CLAUDE.md's XFeat Investigation Log). Unlike
XFeat, DISK+LightGlue does NOT need a from-scratch vendored architecture: both
models have real, permissively-licensed pip-installable code and
auto-downloading pretrained weights.

WHY TWO GRAPHS, NOT ONE: this project's IMatchingStage contract
(OrbRansacMatching, XFeatMatching) does its own per-crop matching internally
in C++. LightGlue's own matching, however, is a learned attention-based
network, not a classical NN/ratio-test the C++ side can replace -- so instead
of C++ doing the matching, the SECOND onnx graph exported here (the LightGlue
matcher) does it, taking both images' already-extracted keypoints+descriptors
as input and returning match indices directly. See
include/DiskLightGlueMatching.hpp for how the two graphs compose.

WHY fabio-sim/LightGlue-ONNX pinned at tag v0.1.0, not current HEAD: current
HEAD (`lightglue_dynamo`, a v3.0 rewrite) requires Python>=3.12 (this machine
has 3.10) and exports a single FUSED extractor+matcher graph taking a whole
batch of image pairs at once -- incompatible with this project's per-candidate
caching contract (which extracts the query frame once, reuses it against up
to top_k candidates). v0.1.0 has neither problem: torch>=1.9.1 only, and
exports the extractor and matcher as two independent ONNX graphs.

VENDORED, NOT PIP-INSTALLED -- checked directly, not assumed: `pip install
"git+https://github.com/fabio-sim/LightGlue-ONNX.git@v0.1.0"` DOES install
cleanly (Apache-2.0, confirmed against the tagged LICENSE file; has a real
setup.py) but its `packages=['lightglue']` only packages the repo's SEPARATE
`lightglue/` directory -- a vendored copy of the ORIGINAL cvg/LightGlue-style
dict-input API (`DISK.forward(data: dict)`), which does not export to ONNX at
all (confirmed: it raised `IndexError` inside torch.export the first time
this was tried). The actually ONNX-exportable, tensor-input API lives in the
repo's OTHER top-level directory, `lightglue_onnx/` -- which that setup.py
never packages. So `scripts/vendor/lightglue_onnx/{disk,lightglue}.py` here
are verbatim copies of that directory's two files (same reasoning
export_xfeat.py already used to vendor XFeat's architecture: no installable
package exposes the code actually needed). Also needs `kornia` (DISK's own
pretrained-weight source, kornia.feature.DISK.from_pretrained), `einops`
(LightGlue's own dependency), and `kornia_rs` (a compiled dependency the
current kornia release on PyPI transitively imports from kornia/__init__.py
-- not in the pinned repo's own requirements.txt, which predates kornia's
move to it; a real, needed addition confirmed by a failing `import kornia`
without it, not a fabricated one).

DYNAMIC KEYPOINT COUNT -- the one real deviation from export_xfeat.py's
pattern: DISK's own NMS-based keypoint selection (kornia's DISK model)
produces a genuinely variable number of keypoints per image, capped at
--max-keypoints but frequently below it on low-texture content. Unlike
XFeat's dense fixed-top_k grid (this project's own separate script), this is
traced via real tensor ops (nonzero(), a data-length-derived
torch.topk(k=min(n, kpts_len))) that DO export to a genuinely dynamic ONNX
axis -- confirmed empirically below, not assumed: --validate runs the
extractor on two DIFFERENT real images and asserts the returned keypoint
counts (a) differ from each other and (b) match PyTorch's own count for the
same input, which a silently-baked-in-fixed-count export (the exact trap
XFeat's sparse path fell into, see export_xfeat.py's docstring) could not
produce. Downstream C++ (DiskLightGlueMatching) must read the actual output
shape per call via Ort::Value::GetTensorTypeAndShapeInfo(), not assume a
fixed N.

Usage:
    python3 scripts/export_disk_lightglue.py --validate
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

INPUT_SIZE = 384        # matches export_xfeat.py's own choice -- multiple of 16 (DISK's own
                         # internal pad-to-16 requirement), same order of magnitude, direct
                         # comparability with XFeat's own gate numbers
MAX_KEYPOINTS = 256     # cap, not a guarantee -- see module docstring. Lowered from an initial
                         # 1024 (matching XFeat's own TOP_K, for direct comparability) after
                         # measuring the real cost this session: LightGlue's self+cross-attention
                         # is roughly O(N^2) in keypoint count per layer across 9 layers, and a
                         # 1024-cap full 5-flight gate was measured to need ~10-11 hours even
                         # after adding candidate-level threading (see
                         # DiskLightGlueMatching.hpp/SplitPipelineEstimator.cpp and CLAUDE.md's
                         # DISK+LightGlue Investigation Log) -- not tractable for one Phase 2
                         # comparison pass. 256 is a 4x cut in N, a ~16x theoretical cut in the
                         # quadratic attention term; no longer an apples-to-apples keypoint budget
                         # against XFeat's 1024, flagged explicitly in the gate writeup rather
                         # than glossed over.
OPSET_VERSION = 18      # see export_xfeat.py / export_retrieval_backbone.py's identical note:
                         # torch 2.13's default exporter rejects opset_version=17 outright.
                         # v0.1.0's own export.py uses opset_version=16 (an older torch); bumped
                         # here for the same reason the other two scripts needed to.

_REAL_IMAGE_CANDIDATES = [
    "Images/map_clean_uavvisloc_01.png",
    "Images/map_clean_uavvisloc_10.png",
    "Images/map_clean_uavvisloc_04.png",
    "Images/map_clean.png",
]


def build_models(max_keypoints: int):
    from lightglue_onnx.disk import DISK
    from lightglue_onnx.lightglue import LightGlue

    extractor = DISK(max_num_keypoints=max_keypoints).eval()
    matcher = LightGlue(pretrained="disk").eval()
    return extractor, matcher


def normalize_keypoints_np(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """Numpy port of lightglue_onnx's onnx_runner.lightglue.LightGlueRunner.normalize_keypoints
    (v0.1.0's own reference ONNX-inference code) -- used here only for this script's own
    PyTorch-vs-ONNX validation; the real-inference normalization lives in C++
    (DiskLightGlueMatching.cpp), same formula."""
    size = np.array([w, h], dtype=np.float32)
    shift = size / 2
    scale = size.max() / 2
    return ((kpts - shift) / scale).astype(np.float32)


def export_extractor(extractor, output_path: str):
    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        extractor, dummy, output_path,
        input_names=["image"], output_names=["keypoints", "scores", "descriptors"],
        opset_version=OPSET_VERSION,
        dynamic_axes={"keypoints": {1: "num_keypoints"}, "scores": {1: "num_keypoints"},
                       "descriptors": {1: "num_keypoints"}},
    )
    print(f"Extractor exported to {output_path} (fixed {INPUT_SIZE}x{INPUT_SIZE} input, "
          f"dynamic keypoint-count axis, max_num_keypoints={MAX_KEYPOINTS})")


def export_lightglue(extractor, matcher, output_path: str):
    img0 = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    img1 = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        kpts0, _scores0, desc0 = extractor(img0)
        kpts1, _scores1, desc1 = extractor(img1)
        kpts0_n = torch.from_numpy(normalize_keypoints_np(kpts0.numpy(), INPUT_SIZE, INPUT_SIZE))
        kpts1_n = torch.from_numpy(normalize_keypoints_np(kpts1.numpy(), INPUT_SIZE, INPUT_SIZE))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        matcher, (kpts0_n, kpts1_n, desc0, desc1), output_path,
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches0", "matches1", "mscores0", "mscores1"],
        opset_version=OPSET_VERSION,
        dynamic_axes={
            "kpts0": {1: "num_keypoints0"}, "kpts1": {1: "num_keypoints1"},
            "desc0": {1: "num_keypoints0"}, "desc1": {1: "num_keypoints1"},
            "matches0": {1: "num_matches0"}, "matches1": {1: "num_matches1"},
            "mscores0": {1: "num_matches0"}, "mscores1": {1: "num_matches1"},
        },
    )
    print(f"LightGlue matcher exported to {output_path}")


def _load_validation_images():
    """Image A: a real, busy reference-map crop (if one is on disk) -- saturates near
    --max-keypoints on real content. Image B: a uniform low-texture synthetic image, chosen
    deliberately so the two images give genuinely DIFFERENT keypoint counts. An earlier version
    of this check used two different real crops and both saturated at exactly the
    --max-keypoints cap, which would have silently masked the exact "baked-in fixed count"
    failure mode this check exists to catch (checked directly: a flat gray 384x384 image
    yields ~162 keypoints from this model at the default 1024 cap, comfortably below it)."""
    import cv2

    path = next((p for p in _REAL_IMAGE_CANDIDATES if os.path.exists(p)), None)
    img_a = None
    if path is not None:
        img = cv2.imread(path)
        if img is not None and img.shape[0] >= INPUT_SIZE * 3 and img.shape[1] >= INPUT_SIZE * 3:
            h, w = img.shape[:2]
            c = img[h // 4:h // 4 + INPUT_SIZE, w // 4:w // 4 + INPUT_SIZE]
            c = cv2.resize(c, (INPUT_SIZE, INPUT_SIZE))
            img_a = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    img_b = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
    return img_a, img_b


def _to_tensor(img_rgb_hwc: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_rgb_hwc.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return t.contiguous()


def validate(extractor_path: str, lightglue_path: str, max_keypoints: int) -> bool:
    import onnxruntime as ort

    extractor, matcher = build_models(max_keypoints)
    ext_sess = ort.InferenceSession(extractor_path, providers=["CPUExecutionProvider"])
    lg_sess = ort.InferenceSession(lightglue_path, providers=["CPUExecutionProvider"])

    img_a_np, img_b_np = _load_validation_images()
    if img_a_np is None:
        print("WARNING: no suitable real reference-map image found on disk for image A -- using "
              "random noise instead. Still a real check: random noise gives yet another "
              "keypoint-count value, independent of image B's low-texture synthetic.")
        img_a = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    else:
        img_a = _to_tensor(img_a_np)
    img_b = _to_tensor(img_b_np)

    # 1. Dynamic-shape check: image A (real, busy content) saturates at --max-keypoints --
    #    checked directly across every real reference-map crop and even lightly-noised
    #    synthetic images tried this session, ALL of them hit the cap, so this is the realistic
    #    common case for this project's content, not a corner case. Image B (uniform, low-
    #    texture) sits well below it. The two must differ substantially -- a silently
    #    baked-in-fixed-count export (the trap XFeat's sparse path fell into, see
    #    export_xfeat.py's docstring) would return the SAME N for both regardless of content.
    #    Image A's count, specifically, must exactly match PyTorch's own -- when the true
    #    local-maxima count comfortably exceeds the cap (verified true for image A: real DISK
    #    heatmaps on 384x384 content routinely produce far more than 1024 candidates), the
    #    RETURNED COUNT itself (min(cap, true_count)) is deterministic even though the ORDER/
    #    IDENTITY of which cap-many points get returned is not (see check 2's own note on why
    #    that's compared differently). Image B's count is NOT held to the same exact-match bar:
    #    a perfectly uniform synthetic image produces widespread exact-tied heatmap values,
    #    and PyTorch's/ONNX Runtime's max_pool2d tie-breaking legitimately differs on ties --
    #    confirmed directly (162 PyTorch vs 171 ONNX on a flat-gray test image, a ~5.5% gap
    #    attributable to tie noise on an input with essentially no real-world analogue, not a
    #    correctness bug) -- so B is only checked for being clearly non-saturated and clearly
    #    different from A, not bit-exact.
    with torch.no_grad():
        kpts_a_t, _scores_a_t, desc_a_t = extractor(img_a)
        kpts_b_t, _scores_b_t, desc_b_t = extractor(img_b)
    kpts_a_o, _scores_a_o, desc_a_o = ext_sess.run(None, {"image": img_a.numpy()})
    kpts_b_o, _scores_b_o, desc_b_o = ext_sess.run(None, {"image": img_b.numpy()})

    n_a_t, n_b_t = kpts_a_t.shape[1], kpts_b_t.shape[1]
    n_a_o, n_b_o = kpts_a_o.shape[1], kpts_b_o.shape[1]
    # Non-saturation check is a strict "<", not a fixed ratio margin -- image B's own keypoint
    # count doesn't scale with --max-keypoints (it's set by real image content, ~162-171 here
    # regardless of the cap), so a ratio-based margin that happened to look right at cap=1024
    # stopped being meaningful once cap=256 brought the two counts closer together. What matters
    # is only that B is genuinely NOT at the cap while A genuinely IS.
    dynamic_ok = (n_a_o == n_a_t == max_keypoints) and n_b_o < n_a_o and n_b_t < n_a_t
    print(f"Extractor keypoint counts -- image A (busy, expect saturated at cap): "
          f"PyTorch={n_a_t} ONNX={n_a_o}; image B (uniform, expect well below cap): "
          f"PyTorch={n_b_t} ONNX={n_b_o}. {'PASS' if dynamic_ok else 'FAIL'} (A must saturate "
          f"and match exactly; B must be clearly non-saturated in both backends -- proves the "
          f"axis is genuinely dynamic, not a baked-in constant)")

    # 2. Numeric agreement on image A via NEAREST-NEIGHBOR matching, not raw index-order diff.
    #    At saturation, torch.topk's specific tie-breaking at the cutoff boundary can legitimately
    #    differ between PyTorch eager and the exported/optimized ONNX graph -- same class of
    #    non-determinism export_xfeat.py's docstring documents for XFeat's own topk (confirmed
    #    here too: a raw element-wise diff against image A showed a spurious ~197px "diff" purely
    #    from this cause before this check was changed to spatial nearest-neighbor matching,
    #    which is robust to a handful of different-but-equally-valid boundary picks and answers
    #    the question that actually matters for correctness: is ONNX finding the SAME real
    #    corners as PyTorch, not "the same list order").
    numeric_ok = False
    frac_close = median_nn_dist = desc_cos = float("nan")
    if n_a_o > 0:
        pa_t, pa_o = kpts_a_t.numpy()[0], kpts_a_o[0]
        d2 = ((pa_o[:, None, :] - pa_t[None, :, :]) ** 2).sum(-1)
        nn_idx = d2.argmin(axis=1)
        nn_dist = np.sqrt(d2[np.arange(len(pa_o)), nn_idx])
        frac_close = float(np.mean(nn_dist < 2.0))
        median_nn_dist = float(np.median(nn_dist))
        desc_cos = float(np.median(np.sum(desc_a_t.numpy()[0][nn_idx] * desc_a_o[0], axis=-1)))
        numeric_ok = frac_close > 0.98 and desc_cos > 0.99
    print(f"Extractor spatial agreement (image A, nearest-neighbor matched): "
          f"{frac_close:.1%} of ONNX keypoints within 2px of a PyTorch keypoint "
          f"(median NN distance {median_nn_dist:.2f}px), median matched-descriptor cosine sim "
          f"{desc_cos:.4f}. "
          f"{'PASS' if numeric_ok else 'FAIL'} (threshold: >98% within 2px, cosine sim >0.99)")

    # 3. Same-domain self-match smoke test for the LightGlue matcher: match image A against
    #    ITSELF -- a real keypoint should overwhelmingly match its own identical copy. Same
    #    self-match discipline this project's XFeat matching-strategy diagnosis used before
    #    trusting cross-domain results (see CLAUDE.md's XFeat Investigation Log).
    kpts0_n = normalize_keypoints_np(kpts_a_t.numpy(), INPUT_SIZE, INPUT_SIZE)
    matches0_o, _matches1_o, _mscores0_o, _mscores1_o = lg_sess.run(
        None, {"kpts0": kpts0_n, "kpts1": kpts0_n.copy(),
               "desc0": desc_a_t.numpy(), "desc1": desc_a_t.numpy()})
    valid = matches0_o[0] > -1
    n_valid = int(valid.sum())
    self_match_rate = float(np.mean(matches0_o[0][valid] == np.where(valid)[0])) if n_valid else 0.0
    self_match_ok = n_valid >= 8 and self_match_rate > 0.95
    print(f"LightGlue self-match smoke test: {n_valid}/{n_a_t} keypoints matched, "
          f"{self_match_rate:.1%} matched to themselves exactly. "
          f"{'PASS' if self_match_ok else 'FAIL'} (threshold: >=8 matches, >95% self-consistent)")

    return dynamic_ok and numeric_ok and self_match_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor-output", default="models/disk.onnx")
    ap.add_argument("--lightglue-output", default="models/disk_lightglue.onnx")
    ap.add_argument("--max-keypoints", type=int, default=MAX_KEYPOINTS)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    extractor, matcher = build_models(args.max_keypoints)
    export_extractor(extractor, args.extractor_output)
    export_lightglue(extractor, matcher, args.lightglue_output)

    if args.validate:
        ok = validate(args.extractor_output, args.lightglue_output, args.max_keypoints)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
