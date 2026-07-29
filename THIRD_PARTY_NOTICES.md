# Third-Party Notices

MerMED-FM builds on several open-source projects. This file records what was
reused, where it lives in this repository, and the terms it arrived under.

The repository as a whole is released under **CC BY-NC 4.0** (see [`LICENSE`](LICENSE)).
That choice is not arbitrary: the finetuning code descends from MAE and RETFound,
both of which are CC BY-NC 4.0, so the non-commercial restriction is inherited and
cannot be relicensed away.

## Components

| Project | Where it is used here | License |
|---|---|---|
| [MAE](https://github.com/facebookresearch/mae) (Meta) | `finetuning/main_finetune*.py`, `finetuning/main_external.py`, `finetuning/engine_finetune*.py`, `finetuning/util/{misc,lr_sched,pos_embed}.py` | CC BY-NC 4.0 |
| [RETFound](https://github.com/rmaphoh/RETFound_MAE) | same files as MAE — the finetuning pipeline is adapted from RETFound's adaptation of MAE | CC BY-NC 4.0 |
| [DINO](https://github.com/facebookresearch/dino) (Meta) | `pretraining/utils.py`, `finetuning/visualize_attention.py`; the teacher–student pretraining objective | Apache-2.0 |
| [DINOv2](https://github.com/facebookresearch/dinov2) (Meta) | `pretraining/koleo.py` (KoLeo regularizer) | Apache-2.0 |
| [iBOT](https://github.com/bytedance/ibot) (ByteDance) | `pretraining/models/vision_transformer.py` | Apache-2.0 |
| [MaSSL](https://github.com/sthalles/MaSSL) | `pretraining/{memory_bank,random_partition,criterion,head}.py` and parts of `pretraining/main_mermed.py` — memory bank and random prototype partitioning | **No license published** — see below |
| [timm](https://github.com/huggingface/pytorch-image-models) | ViT/Swin backbone builders in `finetuning/models_vit.py`, and a dependency throughout | Apache-2.0 |

## Note on MaSSL

MerMED's memory bank and random prototype partitioning are adapted from
**MaSSL** ("Learning from Memory: A Non-Parametric Memory Augmented
Self-Supervised Learning of Visual Features", Silva et al.).

At the time of this release, <https://github.com/sthalles/MaSSL> publishes **no
LICENSE file and declares no license**. Absent an explicit grant, the reuse terms
for the MaSSL-derived components in `pretraining/` are undetermined — an
unlicensed public repository does not by itself convey permission to copy,
modify, or redistribute.

We attribute MaSSL in full here and in the affected source files. Anyone
intending to redistribute or build on those specific components should contact
the MaSSL authors directly to confirm terms. If MaSSL later publishes a license,
this file should be updated to reflect it.

## Files carrying upstream headers

The following retain their original copyright headers. Several reference "the
LICENSE file in the root directory of this source tree" — for those files, the
governing terms are the upstream project's as listed above, not this
repository's `LICENSE`:

- `pretraining/koleo.py` — Meta Platforms, Inc. (DINOv2)
- `pretraining/utils.py` — Facebook, Inc. (DINO, Apache-2.0 header inline)
- `pretraining/models/vision_transformer.py` — ByteDance, Inc. (iBOT)
- `finetuning/visualize_attention.py` — Facebook, Inc. (DINO)
- `finetuning/main_finetune.py`, `finetuning/main_finetune_multilabel.py`,
  `finetuning/main_finetune_fairness.py`, `finetuning/main_external.py`,
  `finetuning/engine_finetune.py`, `finetuning/engine_finetune_multilabel.py`,
  `finetuning/engine_finetune_fairness.py`, `finetuning/util/misc.py`,
  `finetuning/util/lr_sched.py`, `finetuning/util/pos_embed.py`
  — Meta Platforms, Inc. (MAE), revised via RETFound

## Pretrained weights

The released `MerMED.pth` checkpoint is distributed separately at
<https://huggingface.co/youngzhou12/MerMED> and is subject to the license stated
on that model card.
