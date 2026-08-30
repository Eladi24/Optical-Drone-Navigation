# Vendored: NCNet neighbourhood-consensus modules

From **[ignacio-rocco/ncnet](https://github.com/ignacio-rocco/ncnet)** (MIT, © 2018 Ignacio Rocco),
paper *"Neighbourhood Consensus Networks"* (Rocco et al., NeurIPS 2018).

**Why vendored** (not pip-installed): the upstream repo is not a package — it's a research repo with a
training harness, its own ResNet-101 backbone, and `lib.`-relative imports. STRATEGY.md Phase 2+
**Option A2** needs only the backbone-independent consensus pieces, fed our own fine-tuned DINOv2 dense
patch tokens instead of NCNet's ResNet features. This is the same vendoring pattern the project already
uses for `scripts/vendor/lightglue_onnx/`.

**What's here** (kept byte-for-byte against upstream where possible, so it can be diffed):
- `conv4d.py` — `Conv4d` (a loop of `F.conv3d` over the 4th spatial dim; the loop range is fixed at
  the grid size, so it unrolls at ONNX trace time). Verbatim from `lib/conv4d.py`.
- `ncnet_consensus.py` — `featureL2Norm`, `FeatureCorrelation` (4D path), `NeighConsensus` (the 3×
  Conv4d + ReLU stack, symmetric mode), `MutualMatching` (the soft mutual-NN normalization). Lifted
  from `lib/model.py`; the `import` line points at the sibling `conv4d` instead of `lib.conv4d`, and
  the docstrings are lightly trimmed — otherwise unchanged.
- `LICENSE` — upstream MIT license.

**Not vendored:** `FeatureExtraction`, `ImMatchNet`, `fpn_body`, `maxpool4d`, `lib/torch_util`, the
training/eval scripts, the ResNet checkpoints.

**Pretrained consensus weights** (`ncnet_pfpascal.pth.tar`, PF-Pascal semantic correspondence, indoor)
are NOT vendored — Option A2 Stage A2.0 tests the *mechanism* with random init first, and only fetches
those weights (a few MB, MIT) if the mechanism shows promise; Stages A2.1+ retrain the module on
UAV-VisLoc regardless.
