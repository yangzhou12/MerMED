# MerMED-FM — Finetuning & Evaluation

Finetune the released MerMED-FM ViT-B/16 backbone on downstream medical-imaging
classification tasks and evaluate it.

See the [root README](../README.md) for installation, how to obtain
`../weights/MerMED.pth`, the label-CSV format, the full commands, and citation.

## Entry points

| Script | Use |
|--------|-----|
| `main_finetune.py` | Standard single-label finetuning + evaluation (all 2-D modalities). |
| `main_finetune_multilabel.py` | Multi-label tasks (BCE loss, macro/micro metrics). |
| `main_finetune_fairness.py` | Finetuning with subgroup fairness evaluation (`--sensitive_attr`). |
| `main_external.py` | Train on one dataset, evaluate on an external test set. |
| `grad_cam.py` | Grad-CAM heatmaps for a finetuned checkpoint. |
| `visualize_attention.py` | Self-attention map visualization. |
| `scripts/finetune_mermed.sh` | Batch launcher: train sizes × seeds × datasets. |

All entry points build the backbone through `models_vit` and load MerMED via
`--finetune`. The loader handles the pretraining checkpoint's `teacher`/`student`
key layout, strips the `module.backbone.` prefix, interpolates position
embeddings, and attaches a fresh classification head.

Each run writes per-epoch `metrics_{val,test}.csv`, per-sample
`outputs_test.{csv,npz}`, calibration and confusion-matrix plots, and
`checkpoint-best.pth` under `--output_dir/<task>/`.
