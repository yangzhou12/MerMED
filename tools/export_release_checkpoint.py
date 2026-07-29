#!/usr/bin/env python3
"""Reduce a MerMED pretraining checkpoint to a distributable release artifact.

Pretraining writes a full training state: student and teacher weights, the
optimizer state, the memory bank, the AMP scaler, and the ``argparse.Namespace``
of the run. Only the teacher weights are needed downstream, and the pickled
Namespace both bloats the file and embeds the training machine's local paths.

This script keeps one weight tree and nothing else, so the result:

  * is roughly 5x smaller than the training checkpoint,
  * contains no local filesystem paths or other run metadata, and
  * loads under ``torch.load(..., weights_only=True)``, since it holds only
    tensors.

Key names are preserved verbatim, so the exported file is a drop-in replacement
wherever the training checkpoint was accepted (see the loader in
``finetuning/main_finetune.py``).

Examples
--------
    # Standard release export: teacher weights only.
    python tools/export_release_checkpoint.py weights/MerMED.pth MerMED_release.pth

    # Backbone only, dropping the SSL projection head.
    python tools/export_release_checkpoint.py weights/MerMED.pth MerMED_vitb16.pth \
        --backbone-only
"""
import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strip a MerMED training checkpoint down to release weights.",
    )
    parser.add_argument("input", help="Training checkpoint (e.g. weights/MerMED.pth)")
    parser.add_argument("output", help="Path to write the exported checkpoint to")
    parser.add_argument(
        "--key",
        default="teacher",
        help="Which weight tree to keep (default: teacher, the released model)",
    )
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Drop the self-supervised projection head, keeping only the ViT backbone",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.force:
        sys.exit(f"refusing to overwrite existing {args.output} (pass --force)")

    # weights_only=False is required here: training checkpoints pickle an
    # argparse.Namespace. Only run this on checkpoints you produced or trust.
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or args.key not in checkpoint:
        available = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        sys.exit(f"{args.input!r} has no {args.key!r} entry (found: {available})")

    state_dict = checkpoint[args.key]
    dropped = sorted(k for k in checkpoint if k != args.key)

    if args.backbone_only:
        kept = {k: v for k, v in state_dict.items() if ".backbone." in k or k.startswith("backbone.")}
        if not kept:
            sys.exit(
                "--backbone-only matched no keys; inspect the checkpoint's key "
                f"names (first few: {list(state_dict)[:3]})"
            )
        removed = len(state_dict) - len(kept)
        state_dict = kept
        print(f"dropped {removed} non-backbone tensor(s)")

    torch.save({args.key: state_dict}, args.output)

    params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    before = os.path.getsize(args.input) / 1e9
    after = os.path.getsize(args.output) / 1e9
    print(f"kept   '{args.key}': {len(state_dict)} tensors, {params / 1e6:.1f}M parameters")
    if dropped:
        print(f"removed {', '.join(repr(k) for k in dropped)}")
    print(f"{args.input} ({before:.2f} GB) -> {args.output} ({after:.2f} GB)")

    # The point of the exercise: no pickled objects left, so the artifact is
    # loadable without trusting its provenance.
    torch.load(args.output, map_location="cpu", weights_only=True)
    print("verified: loads with weights_only=True")


if __name__ == "__main__":
    main()
