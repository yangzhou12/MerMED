# MerMED-FM — Finetuning & Evaluation

Finetune the released MerMED-FM ViT-B/16 backbone on downstream medical-imaging
classification tasks, evaluate it, and aggregate metrics across seeds.

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
| `get_MerMED_*_results.py` | Aggregate per-run metrics across seeds (summary + statistics). |
| `scripts/finetune_mermed.sh` | Batch launcher: train sizes × seeds × datasets. |

All entry points build the backbone through `models_vit` and load MerMED via
`--finetune`. The loader handles the pretraining checkpoint's `teacher`/`student`
key layout, strips the `module.backbone.` prefix, interpolates position
embeddings, and attaches a fresh classification head.

`models_vit.py` also provides builders for the comparison foundation models
reported in the paper (BiomedCLIP, UniMed-CLIP, UNI, DINOv2, Swin), so they can be
run through the same pipeline.

## Aggregation scripts

Each script lists the methods and datasets to compare in module-level variables at
the top of the file — edit them to match the runs present under `--result_dir`.
The multi-label aggregator instead takes its datasets on the command line:

```bash
python get_MerMED_multilabel_results.py --result_dir <RESULTS_DIR> --datasets my_cxr:5
```
