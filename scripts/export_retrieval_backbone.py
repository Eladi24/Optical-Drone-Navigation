#!/usr/bin/env python3
"""
Exports a small, frozen, ImageNet-pretrained image backbone to ONNX for use as
STRATEGY.md Phase 2's learned retrieval embedding (see CLAUDE.md's "Investigation
Log: STRATEGY.md Phase 2, Learned Retrieval" for the full context). No training
or fine-tuning happens here -- Phase 2 is explicitly "frozen inference only."

Default backbone is DeiT-Tiny-Distilled (5M params, 192-dim embedding) --
STRATEGY.md cites NaviLoc's ablation showing it beating DINOv2-ViT-G/14 (1.1B
params) by 8x on this exact cross-view retrieval task (19.5m vs 162m). DINOv2-S
(21M params, 384-dim) is kept available via --backbone as the comparison point
STRATEGY.md asks for, not as the default.

Embedding readout: with num_classes=0 (classification head stripped), calling
the timm model directly already returns its architecture-appropriate pooled
pre-logits representation -- verified directly against
forward_head(forward_features(x), pre_logits=True), which they're identical
for both backbones tested. For DeiT-Tiny-Distilled specifically this matters:
naively slicing token index 0 (a bare CLS token) would silently discard the
distillation token teacher-signal that's the entire point of the "distilled"
variant -- timm's own pooling for this architecture averages CLS and
distillation tokens instead, so using the model's own forward() (num_classes=0)
rather than hand-rolled token slicing is both simpler and more correct.

The exported graph takes a fixed [1, 3, 224, 224] input (no dynamic axes --
this pipeline embeds one crop/frame at a time, never batched, and a fixed
shape is the simplest, most portable path for ONNX Runtime's CPU execution
provider) and outputs the raw (not L2-normalized) embedding; normalization is
left to the C++ retrieval stage, matching where every other per-frame
normalization step already lives in this codebase.

Requires torch/timm/onnx/onnxscript in the project's Python venv (../CV_IP/cv_env,
sibling to this repo -- NOT the bare system python3). Install once with:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install timm onnx onnxruntime onnxscript
(torch and torchvision must come from the SAME --index-url -- pulling
torchvision from default PyPI produced a real ABI mismatch this session,
"operator torchvision::nms does not exist"; see CLAUDE.md.)

Usage:
    python3 scripts/export_retrieval_backbone.py --backbone deit_tiny \
        --output models/deit_tiny_retrieval.onnx --validate

    python3 scripts/export_retrieval_backbone.py --backbone dinov2_s \
        --output models/dinov2_s_retrieval.onnx --validate  # comparison point, not default

    # A fine-tuned checkpoint (scripts/finetune_retrieval_backbone.py's output, a plain
    # state_dict .pt matching this same architecture) -- --weights replaces the stock
    # ImageNet pretrained=True load with pretrained=False + load_state_dict, mirroring
    # export_xfeat.py's build_model()/--weights-cache pattern for loading an arbitrary
    # local checkpoint rather than a package-provided one.
    python3 scripts/export_retrieval_backbone.py --backbone dinov2_s \
        --weights checkpoints/dinov2_s_finetuned.pt \
        --output models/dinov2_s_finetuned_retrieval.onnx --validate
"""
import argparse
import os
import sys

BACKBONES = {
    "deit_tiny": "deit_tiny_distilled_patch16_224",
    "dinov2_s": "vit_small_patch14_dinov2",
}

INPUT_SIZE = 224
# torch 2.13's default (dynamo-based) exporter rejected opset_version=17 outright
# ("Target opset version 17 is not supported. Supported range: 18 to 25") --
# confirmed directly this session, not assumed from documentation.
OPSET_VERSION = 18


class _DensePatchTokens:
    """Wraps a timm ViT so forward() returns its per-patch token grid
    [B, n_patches, C] (prefix/CLS/distill/register tokens dropped) instead of the
    pooled embedding. This is the readout STRATEGY.md's Phase 2+ semantic/dense-
    matching direction needs (arXiv:2506.09748): dense DINOv2 features matched with
    a correlation volume, not a single global vector. Same backbone, same weights
    (incl. a fine-tuned checkpoint) -- only the output head differs, so it lives
    here rather than in a second export script.
    """
    def __init__(self, model):
        import torch.nn as nn
        self._nn = nn
        self.model = model
        self.n_prefix = int(getattr(model, "num_prefix_tokens", 1))

    def make(self):
        nn = self._nn
        model, n_prefix = self.model, self.n_prefix

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x):
                feats = self.m.forward_features(x)      # [B, n_prefix + n_patch, C]
                return feats[:, n_prefix:, :].contiguous()

        w = Wrap()
        w.eval()
        return w


def build_model(backbone_key: str, weights_path: str = None, dense: bool = False):
    import timm

    model_name = BACKBONES[backbone_key]
    # weights_path set -> start from a bare (untrained) architecture and load our own
    # checkpoint instead of timm's stock ImageNet weights; pretrained=False here is not
    # "no pretraining happened", it's "don't let timm silently overwrite the fine-tuned
    # weights we're about to load with its own stock ones".
    kwargs = {"pretrained": weights_path is None, "num_classes": 0}
    if backbone_key == "dinov2_s":
        # DINOv2's native patch14 default img_size is 518; force 224 so every
        # backbone in this script shares one fixed input size end-to-end.
        kwargs["img_size"] = INPUT_SIZE
    model = timm.create_model(model_name, **kwargs)
    if weights_path is not None:
        import torch
        state_dict = torch.load(weights_path, map_location="cpu")
        result = model.load_state_dict(state_dict, strict=True)
        print(f"Loaded fine-tuned weights from {weights_path} ({result})")
    model.eval()
    if dense:
        model = _DensePatchTokens(model).make()
    return model, model_name


def export(backbone_key: str, output_path: str, weights_path: str = None,
           dense: bool = False):
    import torch

    model, model_name = build_model(backbone_key, weights_path, dense=dense)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    out_name = "patch_tokens" if dense else "embedding"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=[out_name],
        opset_version=OPSET_VERSION,
        dynamic_axes=None,
    )

    print(f"Backbone: {model_name} ({backbone_key}){'  [dense patch tokens]' if dense else ''}")
    print(f"Parameters: {n_params:,}")
    if dense:
        _, n_tok, c = out.shape
        g = int(round(n_tok ** 0.5))
        print(f"Output '{out_name}': [1, {n_tok}, {c}]  (~{g}x{g} patch grid, not L2-normalized)")
    else:
        print(f"Embedding dim: {out.shape[-1]}")
    print(f"Input shape: [1, 3, {INPUT_SIZE}, {INPUT_SIZE}] (fixed, no dynamic axes)")
    print(f"Exported to: {output_path}")
    return output_path


def validate(backbone_key: str, output_path: str, weights_path: str = None,
             dense: bool = False, n_samples: int = 5, atol: float = 1e-4):
    import numpy as np
    import onnxruntime as ort
    import torch

    model, _ = build_model(backbone_key, weights_path, dense=dense)
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out_name = "patch_tokens" if dense else "embedding"

    max_diff = 0.0
    for _ in range(n_samples):
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            torch_out = model(x).numpy()
        onnx_out = session.run([out_name], {"image": x.numpy()})[0]
        diff = float(np.max(np.abs(torch_out - onnx_out)))
        max_diff = max(max_diff, diff)

    ok = max_diff < atol
    print(f"Validation ({n_samples} random samples): max |PyTorch - ONNX Runtime| "
          f"diff = {max_diff:.2e} ({'PASS' if ok else 'FAIL'}, atol={atol:.0e})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default="deit_tiny")
    ap.add_argument("--output", required=True, help="Output .onnx path (e.g. models/deit_tiny_retrieval.onnx)")
    ap.add_argument("--weights", default=None,
                     help="Path to a fine-tuned state_dict .pt (scripts/finetune_retrieval_backbone.py's "
                          "output) matching --backbone's architecture. Omit to export timm's stock "
                          "ImageNet-pretrained weights (the existing frozen-baseline behavior, unchanged).")
    ap.add_argument("--validate", action="store_true",
                     help="Reload the exported graph via onnxruntime and diff against live PyTorch output")
    ap.add_argument("--dense", action="store_true",
                     help="Export the per-patch token grid [1, n_patch, C] (output 'patch_tokens') "
                          "instead of the pooled retrieval embedding -- for STRATEGY.md's Phase 2+ "
                          "dense/semantic matching direction (DinoDenseMatching). Same weights, "
                          "different readout.")
    args = ap.parse_args()

    export(args.backbone, args.output, args.weights, dense=args.dense)

    if args.validate:
        ok = validate(args.backbone, args.output, args.weights, dense=args.dense)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
