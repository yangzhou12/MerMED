# MerMED-FM

**Multimodal, Multi-Disease Medical Imaging Foundation Model**

[![Paper](https://img.shields.io/badge/Lancet%20Digital%20Health-10.1016%2Fj.landig.2026.101007-b31b1b)](https://doi.org/10.1016/j.landig.2026.101007)
[![arXiv](https://img.shields.io/badge/arXiv-2507.00185-b31b1b)](https://arxiv.org/abs/2507.00185)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-MerMED-yellow)](https://huggingface.co/youngzhou12/MerMED)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

MerMED-FM is a self-supervised foundation model for medical imaging covering **seven
modalities**: color fundus photography (CFP), optical coherence tomography (OCT),
chest X-ray (CXR), CT, histopathology, ultrasound, and dermatology.

This repository holds the pretraining and finetuning code. For the method, datasets,
and results, see the [paper](https://doi.org/10.1016/j.landig.2026.101007).

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

The checkpoint is distributed via HuggingFace (1.6 GB):
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

`MerMED.pth` is the **teacher** backbone at the epoch-50 snapshot of the 100-epoch
pretraining schedule. It is a full training checkpoint with keys `student`, `teacher`,
`optimizer`, `memory_bank`, `epoch`, `args`, `fp16_scaler`; downstream code reads
`teacher`.

> Loading it requires `torch.load(..., weights_only=False)`, because it pickles an
> `argparse.Namespace`. Only load checkpoints you trust.
> `python tools/export_release_checkpoint.py weights/MerMED.pth out.pth` reduces it to
> a tensors-only file (1.61 GB → 0.37 GB) that loads under `weights_only=True`.

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

**Finetuning** instead uses `--global_pool avg` and deliberately reinitializes the
final norm so it trains with the new classification head. Use the snippet above for
frozen embeddings.

## Repository layout

```
MerMED/
├── pretraining/    # self-supervised pretraining — see pretraining/README.md
├── finetuning/     # finetuning, evaluation, explainability — see finetuning/README.md
├── tools/          # export_release_checkpoint.py
└── weights/        # place MerMED.pth here (git-ignored)
```

## Data format

No data ships with this repository. Both stages read CSV manifests, with image paths
resolved against a data root you supply.

**Pretraining manifest** — columns `image_id, image_path, modality`:

```csv
image_id,image_path,modality
TRAIN032990.jpg,<DATA_ROOT>/JustRAIGS/images/TRAIN032990.jpg,cfp
```

`modality` selects the per-modality mean/std normalization and is matched
**case-sensitively**:

| Value | Modality |
|-------|----------|
| `cfp` | Color fundus photography |
| `oct` | Optical coherence tomography |
| `cxr` | Chest X-ray |
| `CT` | Computed tomography |
| `US` | Ultrasound |
| `pathology` | Histopathology |
| `skin` | Dermatology |
| `eye` | Generic ophthalmic fallback (`cfp` + `oct` pooled statistics) |

Note the capitalization of `CT` and `US`, and that histopathology is `pathology`, not
`path`. An unrecognized value raises a `KeyError` during pretraining. The finetuning
entry points take the same names through `--modality`.

**Downstream labels** — each dataset directory holds a `finetune_labels.csv` with
columns `image_id, image_path, label, split`, where `split ∈ {train, val, test}` and
`image_path` is relative to that dataset's data root:

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

The equivalent explicit command, with the released checkpoint's hyperparameters:

```bash
torchrun --nproc-per-node 8 main_mermed.py \
    --arch vit_base --patch_size 16 --batch_size_per_gpu 128 --num_workers 10 \
    --local_crops_number 10 --global_crops_scale 0.2 1 --local_crops_scale 0.05 0.2 \
    --out_dim 131072 --partition_size 16384 --optimizer adamw --lr 5e-5 --min_lr 1e-06 \
    --weight_decay 0.04 --weight_decay_end 0.4 --layer_decay 0.8 --warmup_epochs 10 \
    --momentum_teacher 0.9995 --warmup_teacher_temp 0.04 --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 10 --drop_path_rate 0.1 --use_bn_in_head true \
    --clip_grad 1 --epochs 100 \
    --data_path <path/to/manifest.csv> \
    --output_dir ./output_mermed
```

Checkpoints are written to `--output_dir` as `checkpoint.pth`, plus
`checkpoint<epoch>.pth` every `--saveckp_freq` epochs. Training is tracked with
[Weights & Biases](https://wandb.ai) by default; pass **`--no_wandb`** to train
without it (checkpointing is unaffected).

> **Batch size constraint.** The memory bank enqueues one slot per sample and requires
> its capacity to divide evenly by the global batch, so
> `--out_dim % (--batch_size_per_gpu × num_gpus)` must be `0`. The released recipe
> satisfies this (`131072 % (128 × 8) == 0`); values such as 96 or 100 per GPU do not,
> and trip an assertion on the first step. Prefer powers of two.

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

- `--task` must match the dataset's directory name under `--data_path`.
- `--train_size` is the fraction of the training split to keep, taken as a
  **stratified subsample** (any float in `(0, 1]`).
- The test split is evaluated at the end of every run. Add `--eval` to evaluate an
  existing checkpoint without training.
- Each run writes `metrics_val.csv`, `metrics_test.csv`, `outputs_test.{csv,npz}`,
  reliability and confusion-matrix plots, and `checkpoint-best.pth`. TensorBoard
  events go to `--log_dir`.

**Variants:** `main_finetune_multilabel.py` (multi-label, BCE loss),
`main_finetune_fairness.py` (subgroup fairness via `--sensitive_attr`),
`main_external.py` (train on one dataset, evaluate on an external test set).

**Batch runs:** edit the CONFIG block in `finetuning/scripts/finetune_mermed.sh`, then
`bash scripts/finetune_mermed.sh`. It loops train sizes × seeds × datasets, allocating
a free GPU per job, and is also the reference list of dataset names and class counts.

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

## License

Released under **[CC BY-NC 4.0](LICENSE)** — use, share and adapt for
**non-commercial** purposes with attribution. Third-party components are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

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
  doi       = {10.1016/j.landig.2026.101007},
  url       = {https://doi.org/10.1016/j.landig.2026.101007},
  publisher = {Elsevier},
  note      = {Published online July 27, 2026},
}
```

## Acknowledgements

MerMED-FM builds on these projects:

- [MaSSL](https://github.com/sthalles/MaSSL) — memory bank and random prototype partitioning
- [MAE](https://github.com/facebookresearch/mae) and [RETFound](https://github.com/rmaphoh/RETFound_MAE) — finetuning pipeline
- [timm](https://github.com/huggingface/pytorch-image-models) — backbone builders

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for per-file attribution.
