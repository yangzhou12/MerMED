# MerMED-FM — Pretraining

Self-supervised pretraining for MerMED-FM: a DINO-style teacher–student objective
extended with a feature/label **memory bank**, **random partitioning** of the
prototype space, and a **KoLeo** regularizer. The teacher backbone produced here
is the released `MerMED.pth` consumed by `../finetuning/`.

See the [root README](../README.md) for installation, the manifest CSV format,
the full training command, and citation.

## Files

| File | Purpose |
|------|---------|
| `main_mermed.py` | Training entry point: arguments, model/optimizer build, train loop. |
| `datasets.py` | `MerMEDDataset`, multi-crop augmentation, `AllClassesImbalancedSampler`. |
| `models/` | ViT backbones (`vit_tiny/small/base/large`). |
| `head.py` | Projection head (MLP → L2-normalized bottleneck). |
| `memory_bank.py` | Cross-rank feature/label memory bank. |
| `random_partition.py` | Random partitioning of the output space. |
| `criterion.py` | Student–teacher cross-entropy loss. |
| `koleo.py` | KoLeo entropic regularizer. |
| `utils.py` | Distributed setup, schedulers, checkpoint I/O, multi-crop wrapper. |
| `train_mermed.sh` | Ready-to-edit launcher. |

## Run

```bash
# Edit the CONFIG block (GPUs, manifest path, output dir, W&B), then:
bash train_mermed.sh
```

Key arguments beyond the hyperparameters:

| Flag | Meaning |
|------|---------|
| `--data_path` | Manifest CSV (`image_id, image_path, modality`). Required. |
| `--output_dir` | Where `checkpoint.pth` and run metadata are written (default `./output_mermed`). |
| `--no_wandb` | Train without Weights & Biases. Checkpointing is unaffected. |
| `--pretrained_path` | Optional backbone checkpoint to initialize from. |
| `--resume_from_dir` | Directory holding a `checkpoint.pth` to resume from. |
| `--saveckp_freq` | Also write `checkpoint<epoch>.pth` every N epochs. |

The released `MerMED.pth` is the **teacher** at the **epoch-50** snapshot of the
100-epoch schedule.

`datasets.py` can be smoke-tested against a manifest directly:

```bash
python datasets.py <path/to/MerMED_Mix4.csv>
```
