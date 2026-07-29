# MerMED-FM

**Multimodal, Multi-Disease Medical Imaging Foundation Model**

[![Paper](https://img.shields.io/badge/Lancet%20Digital%20Health-10.1016%2Fj.landig.2026.101007-b31b1b)](https://doi.org/10.1016/j.landig.2026.101007)
[![arXiv](https://img.shields.io/badge/arXiv-2507.00185-b31b1b)](https://arxiv.org/abs/2507.00185)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-MerMED-yellow)](https://huggingface.co/youngzhou12/MerMED)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

MerMED-FM is a self-supervised foundation model for medical imaging that spans
**seven modalities** and more than ten specialties: color fundus photography (CFP),
optical coherence tomography (OCT), chest X-ray (CXR), CT, histopathology,
ultrasound, and dermatology.

A ViT-B/16 backbone (85.8 M parameters) is pretrained on 3.3 M images with a
DINO-style teacher–student objective, extended with a **feature/label memory
bank**, **random partitioning** of the prototype space, and a **KoLeo**
regularizer. The resulting teacher backbone transfers to downstream
classification by finetuning.

This repository contains the pretraining, finetuning, evaluation, fairness, and
explainability code used in the paper.

### Reported performance

Test AUROC on held-out downstream tasks, by modality:

| OCT | Pathology | Ultrasound | CT | Dermatology |
|:---:|:---:|:---:|:---:|:---:|
| 0.988 | 0.982 | 0.951 | 0.943 | 0.931 |

---

## Contents

- [Installation](#installation)
- [Pretrained weights](#pretrained-weights)
- [Quick start: feature extraction](#quick-start-feature-extraction)
- [Repository layout](#repository-layout)
- [Data format](#data-format)
- [Pretraining](#pretraining)
- [Finetuning and evaluation](#finetuning-and-evaluation)
- [Aggregating results](#aggregating-results)
- [Explainability](#explainability)
- [Data availability](#data-availability)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Installation

Python ≥ 3.10 is required.

```bash
conda create -n mermed python=3.11 -y
conda activate mermed

# Install torch first to control the CUDA build (cu121 shown here).
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/yangzhou12/MerMED.git
cd MerMED
pip install -r requirements.txt
```

## Pretrained weights

The checkpoint is distributed via HuggingFace (1.6 GB, too large for git):
**[youngzhou12/MerMED](https://huggingface.co/youngzhou12/MerMED)**.

```bash
pip install -U "huggingface_hub[cli]"
hf download youngzhou12/MerMED MerMED.pth --local-dir weights
```

or from Python:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("youngzhou12/MerMED", "MerMED.pth")
```

`MerMED.pth` is the **teacher** backbone at the epoch-50 snapshot of the
100-epoch pretraining schedule. It is a full training checkpoint, with keys
`student`, `teacher`, `optimizer`, `memory_bank`, `epoch`, `args`, and
`fp16_scaler`; downstream code reads `teacher`.

> Because the checkpoint pickles an `argparse.Namespace`, loading it requires
> `torch.load(..., weights_only=False)`. Only load checkpoints you trust. To
> produce a smaller, tensors-only artifact that loads safely under
> `weights_only=True`:
>
> ```bash
> python tools/export_release_checkpoint.py weights/MerMED.pth MerMED_release.pth
> ```
>
> This keeps the teacher weights and drops the optimizer, memory bank, student
> copy, and run metadata — 1.61 GB → 0.37 GB.

## Quick start: feature extraction

Extract a 768-dimensional embedding for any medical image. With
`global_pool="token"` every pretrained weight loads exactly, including the final
LayerNorm, so `load_state_dict` reports no missing or unexpected keys:

```python
import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "finetuning")
import models_vit

model = models_vit.vit_base_patch16(
    num_classes=0, global_pool="token", dynamic_img_size=True
)

state = torch.load("weights/MerMED.pth", map_location="cpu", weights_only=False)["teacher"]
state = {k.replace("module.backbone.", ""): v
         for k, v in state.items() if k.startswith("module.backbone.")}
print(model.load_state_dict(state, strict=False))   # missing=[] unexpected=[]
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("your_image.png").convert("RGB")
with torch.no_grad():
    features = model(preprocess(img).unsqueeze(0))   # -> torch.Size([1, 768])
```

Note that **finetuning** instead uses `--global_pool avg` and deliberately
reinitializes the final norm so that it trains together with the new
classification head (see `finetuning/main_finetune.py`). The snippet above is for
frozen embeddings.

## Repository layout

```
MerMED/
├── pretraining/          # self-supervised pretraining (see pretraining/README.md)
│   ├── main_mermed.py        # training entry point
│   ├── datasets.py           # manifest dataset + multi-crop augmentation + sampler
│   ├── models/               # ViT backbones (tiny/small/base/large)
│   ├── head.py               # projection head
│   ├── memory_bank.py        # cross-rank feature/label memory bank
│   ├── random_partition.py   # random partitioning of the output space
│   ├── criterion.py          # student-teacher cross-entropy
│   ├── koleo.py              # KoLeo regularizer
│   └── train_mermed.sh       # ready-to-edit launcher
├── finetuning/           # finetuning, evaluation, explainability (see finetuning/README.md)
│   ├── main_finetune.py           # single-label finetuning + evaluation
│   ├── main_finetune_multilabel.py
│   ├── main_finetune_fairness.py
│   ├── main_external.py           # train on one dataset, test on another
│   ├── models_vit.py              # backbone builders
│   ├── get_MerMED_*_results.py    # metric aggregation across seeds
│   ├── grad_cam.py                # Grad-CAM heatmaps
│   ├── visualize_attention.py     # self-attention maps
│   └── scripts/finetune_mermed.sh # batch launcher
├── tools/
│   └── export_release_checkpoint.py
└── weights/              # place MerMED.pth here (git-ignored)
```

## Data format

No data ships with this repository. Both stages read plain CSV manifests, with
image paths resolved against a data root you supply.

**Pretraining manifest** — columns `image_id, image_path, modality`:

```csv
image_id,image_path,modality
TRAIN032990.jpg,<DATA_ROOT>/JustRAIGS/images/TRAIN032990.jpg,cfp
```

`modality` selects per-modality mean/std normalization; recognized values are
`cfp`, `oct`, `cxr`, `ct`, `us`, `path`, `skin`, and `eye`.

**Downstream labels** — each dataset directory holds a `finetune_labels.csv` with
columns `image_id, image_path, label, split`, where `split ∈ {train, val, test}`
and `image_path` is relative to that dataset's data root:

```csv
image_id,image_path,label,split
Covid (483).png,./COVID/Covid (483).png,Covid-19,train
```

## Pretraining

Edit the CONFIG block in `pretraining/train_mermed.sh` (GPU count, manifest path,
output directory), then:

```bash
cd pretraining
bash train_mermed.sh
```

The equivalent explicit command — these are the hyperparameters recorded in the
released checkpoint:

```bash
torchrun --nproc-per-node 8 main_mermed.py \
    --arch vit_base --patch_size 16 --batch_size_per_gpu 128 --num_workers 10 \
    --local_crops_number 10 --global_crops_scale 0.2 1 --local_crops_scale 0.05 0.2 \
    --out_dim 131072 --partition_size 16384 --optimizer adamw --lr 5e-5 --min_lr 1e-06 \
    --weight_decay 0.04 --weight_decay_end 0.4 --layer_decay 0.8 --warmup_epochs 10 \
    --momentum_teacher 0.9995 --warmup_teacher_temp 0.04 --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 10 --drop_path_rate 0.1 --use_bn_in_head true \
    --clip_grad 1 --epochs 100 \
    --data_path <path/to/MerMED_Mix4.csv> \
    --output_dir ./output_mermed
```

Augmentation produces 2 global crops (224×224) and `--local_crops_number` local
crops (96×96) with per-modality normalization (`DataAugmentationMerMED` in
`main_mermed.py`).

Checkpoints are written to `--output_dir` as `checkpoint.pth`, plus
`checkpoint<epoch>.pth` every `--saveckp_freq` epochs. Training is tracked with
[Weights & Biases](https://wandb.ai) by default; pass **`--no_wandb`** to train
without it (checkpointing is unaffected).

## Finetuning and evaluation

Single dataset:

```bash
cd finetuning
python main_finetune.py \
    --model vit_base_patch16 --global_pool avg --input_size 224 \
    --epochs 50 --batch_size 16 --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 --use_amp \
    --task <DATASET> --nb_classes <N> \
    --data_path <DATA_ROOT>/<DATASET> \
    --label_path <DATA_ROOT>/<DATASET>/finetune_labels.csv \
    --train_size 1.0 --seed 0 \
    --finetune ../weights/MerMED.pth \
    --output_dir <RESULTS_DIR>/MerMED_100_seed0_outputs/ \
    --log_dir   <RESULTS_DIR>/MerMED_100_seed0_logs/
```

- `--train_size ∈ {0.1, 0.3, 0.5, 1.0}` takes a **stratified subsample** of the
  training split for the few-shot regimes.
- The test split is evaluated at the end of every run. Add `--eval` to evaluate an
  existing checkpoint without training.
- Each run writes `metrics_val.csv`, `metrics_test.csv`,
  `outputs_test.{csv,npz}`, reliability and confusion-matrix plots, and
  `checkpoint-best.pth`. TensorBoard events go to `--log_dir`.

**Variants:** `main_finetune_multilabel.py` (multi-label, BCE loss),
`main_finetune_fairness.py` (subgroup fairness via `--sensitive_attr`),
`main_external.py` (train on one dataset, evaluate on an external test set).

**All datasets × seeds × few-shot fractions:** edit the CONFIG block in
`finetuning/scripts/finetune_mermed.sh`, then `bash scripts/finetune_mermed.sh`.
It loops `train_sizes=(0.1 0.3 0.5 1.0)` × `seeds=(0 1 42 123 2025)` × datasets,
allocating a free GPU per job.

Metrics recorded: Accuracy, Balanced Accuracy, AUROC, AUC-PR, Sensitivity,
Specificity, F1, Brier score, ECE, plus per-class variants and per-subgroup
disparities for the fairness runs.

## Aggregating results

```bash
cd finetuning
python get_MerMED_results.py --result_dir <RESULTS_DIR> --output_dir ./aggregated_results
```

Produces `comprehensive_summary.csv` (mean ± confidence interval per metric) and
`auc_f1_statistical_analysis.csv` (t-test / Mann-Whitney comparisons). Variants:
`get_MerMED_per_class_results.py`, `get_MerMED_external_results.py`,
`get_MerMED_multilabel_results.py`, `get_MerMED_fairness_results.py`,
`get_MerMED_ablation_results.py`.

Each script lists the methods and datasets to compare at the top of the file —
edit those to match the runs present under `--result_dir`. The multi-label
aggregator takes them on the command line instead:

```bash
python get_MerMED_multilabel_results.py --result_dir <RESULTS_DIR> --datasets my_cxr:5
```

### Example test performance

MerMED at 100 % training data, one representative dataset per modality. Each
value is the mean across the five seeds (0, 1, 42, 123, 2025):

| Modality | Dataset | # classes | Acc | AUROC | AUC-PR | F1 |
|----------|---------|----------:|----:|------:|-------:|---:|
| CFP (fundus) | JSIEC | 39 | 0.897 | 0.996 | 0.951 | 0.866 |
| OCT | OCTID | 5 | 0.963 | 0.998 | 0.989 | 0.953 |
| Chest X-ray | TBX11K | 3 | 0.994 | 0.999 | 0.999 | 0.993 |
| CT | SARS-COV-2 | 2 | 0.987 | 0.999 | 0.999 | 0.987 |
| Pathology | BreakHis | 2 | 0.994 | 1.000 | 1.000 | 0.993 |
| Ultrasound | BUSI | 3 | 0.862 | 0.973 | 0.944 | 0.843 |
| Dermatology | HAM10000_clean | 7 | 0.932 | 0.987 | 0.922 | 0.850 |

## Explainability

```bash
cd finetuning

# Grad-CAM heatmaps from a finetuned checkpoint
python grad_cam.py --model-path <ckpt> --num-classes <N> \
    --label-path <csv> --image-root <dir> --output-dir ./heatmaps

# Self-attention maps from the pretrained backbone
python visualize_attention.py --arch vit_base_patch16 \
    --pretrained_weights ../weights/MerMED.pth --checkpoint_key teacher \
    --image_path <image> --output_dir ./attention
```

## Data availability

This repository contains no imaging data. The downstream datasets fall into two
groups.

**Publicly available** (obtain from their original sources): APTOS2019, CRFO-v4,
Glaucoma_fundus, IDRiD, JSIEC, MESSIDOR2, PAPILA, OCTDL, OCTID, COVIDx-CXR4,
TBX11K, RSNA Pneumonia, SIIM-ACR Pneumothorax, CBIS-DDSM, chest-ctscan-images,
IQ-OTH/NCCD, SARS-COV-2, HRCTCov19, iCTCF, CRC-VAL-HE-7K, PanNuke,
Kather_Texture_2016, BreakHis, Chaoyang, LC25000, MIDOG25, AMi-Br, BUSC, BUSI,
US3M, BrEaST, BCN20000, Derm7pt, Dermnet, HAM10000, PAD-UFES-20, HIBA, MSKCC, DDI.

**Access-restricted** — available only under a data use agreement or from the
respective data custodians: DRCR_CFP, DRCR_OCT, FM-AMD, FM-CKD, FM-DR,
FM-Glaucoma, FM-MMD, Seed_Cataract, RAPIER_CT, RAPIER_Gastric, TCGA.

The pretraining corpus is assembled from public per-modality image pools; see the
paper for the full provenance.

## License

This repository is released under **[CC BY-NC 4.0](LICENSE)** — free to use,
share, and adapt for **non-commercial** purposes with attribution.

The finetuning code derives from MAE and RETFound, both CC BY-NC 4.0, so the
non-commercial restriction is inherited rather than freely chosen. Some
components carry different terms, and one upstream project publishes no license
at all — see **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)** before
redistributing.

## Citation

If you use MerMED-FM, please cite the paper:

> Zhou Y, Quek CWN, Zhou J, et al. MerMED-FM: Multimodal, Multi-Disease Medical
> Imaging Foundation Model. *Lancet Digit Health*. Published online July 27,
> 2026. doi:10.1016/j.landig.2026.101007

```bibtex
@article{zhou2026mermedfm,
  title     = {MerMED-FM: Multimodal, Multi-Disease Medical Imaging Foundation Model},
  author    = {Zhou, Yang and Quek, Chrystie Wan Ning and Zhou, Jun and Wang, Yan and
               Bai, Yang and Gutierrez, Laura and Ke, Yuhe and Yao, Jie and
               Teo, Zhen Ling and Ting, Darren Shu Jeng and Cheng, Ching-Yu and
               Tham, Yih Chung and Soetikno, Brian T. and Nielsen, Christopher S. and
               Elze, Tobias and Li, Zengxiang and Jao-Yiu Sung, Joseph and
               Li, Kelvin Zhenghao and Hiok Hong, Chan and Ong, Charles Jit Teng and
               Wong, Joy Le Yi and Kuo, Chang-Fu and Wu, We-Chi and
               Ho, Margaret Ming-Chih and Cheng, Lionel Tim-Ee and
               Anh, Tran Nguyen Tuan and Cheng, Chee Leong and Wong, Tien Yin and
               Liu, Nan and Tan, Iain Beehuat and Lim, Tony Kiat Hon and
               Moshfeghi, Darius M. and Goh, Rick Siow Mong and Liu, Yong and
               Ting, Daniel Shu Wei},
  journal   = {The Lancet Digital Health},
  year      = {2026},
  pages     = {101007},
  issn      = {2589-7500},
  doi       = {10.1016/j.landig.2026.101007},
  url       = {https://doi.org/10.1016/j.landig.2026.101007},
  publisher = {Elsevier},
  note      = {Published online July 27, 2026},
}
```

The earlier preprint is [arXiv:2507.00185](https://arxiv.org/abs/2507.00185).

## Acknowledgements

MerMED-FM builds on these projects:

- [MaSSL](https://github.com/sthalles/MaSSL) — memory bank and random prototype partitioning
- [DINO](https://github.com/facebookresearch/dino) — self-distillation objective
- [DINOv2](https://github.com/facebookresearch/dinov2) — KoLeo regularizer
- [iBOT](https://github.com/bytedance/ibot) — ViT implementation
- [MAE](https://github.com/facebookresearch/mae) and [RETFound](https://github.com/rmaphoh/RETFound_MAE) — finetuning pipeline
- [timm](https://github.com/huggingface/pytorch-image-models) — backbone builders
