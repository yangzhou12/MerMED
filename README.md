# MerMED-FM
Official implementation for "[MerMED-FM: Multimodal, Multi-Disease Medical Imaging Foundation Model](https://arxiv.org/abs/2507.00185)"

### 🔧Install environment

1. Create environment with conda:

```
conda create -n retfound python=3.11.0 -y
conda activate retfound
```

2. Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/rmaphoh/RETFound/
cd RETFound
pip install -r requirements.txt
```


### 🌱Fine-tuning with RETFound weights

1. Get access to the pre-trained models on HuggingFace (register an account and fill in the form) and go to step 2:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">Source</th>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureCFP</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureCFP">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureOCT</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureOCT">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_meh">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_shanghai">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_meh">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
</tbody></table>

2. Login in your HuggingFace account, where HuggingFace token can be [created and copied](https://huggingface.co/settings/tokens).
```
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
```

**Optional**: if your machine and server cannot access HuggingFace due to internet wall, run the command below (Do not run it if you can access):
```
export HF_ENDPOINT=https://hf-mirror.com
```

3. If you would like to fine-tune [DINOv2](https://github.com/facebookresearch/dinov2) and [DINOv3](https://github.com/facebookresearch/dinov3), please visit their GitHub repositories to download the model weights and put them in the RETFound folder.

4. Organise your data into this directory structure (Public datasets used in this study can be [downloaded here](BENCHMARK.md))

```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
``` 



5. Start fine-tuning by running `sh train.sh`.


In `train.sh`, the model can be selected by changing the hyperparameters `MODEL`, `MODEL_ARCH`, `FINETUNE`:

**RETFound**:

| MODEL           | MODEL_ARCH               | FINETUNE                 | SIZE                     |
|-----------------|--------------------------|--------------------------|--------------------------|
| RETFound_mae    | retfound_mae             | RETFound_mae_natureCFP   | ~300M                    |
| RETFound_mae    | retfound_mae             | RETFound_mae_natureOCT   | ~300M                    |
| RETFound_mae    | retfound_mae             | RETFound_mae_meh         | ~300M                    |
| RETFound_mae    | retfound_mae             | RETFound_mae_shanghai    | ~300M                    |
| RETFound_dinov2 | retfound_dinov2          | RETFound_dinov2_meh      | ~300M                    |
| RETFound_dinov2 | retfound_dinov2          | RETFound_dinov2_shanghai | ~300M                    |


**DINOv3**:

| MODEL           | MODEL_ARCH               | FINETUNE                         | SIZE                     |
|-----------------|--------------------------|----------------------------------|--------------------------|
| Dinov3          | dinov3_vits16            | dinov3_vits16_pretrain.pth       | ~21M                     |
| Dinov3          | dinov3_vits16plus        | dinov3_vits16plus_pretrain.pth   | ~29M                     |
| Dinov3          | dinov3_vitb16            | dinov3_vitb16_pretrain.pth       | ~86M                     |
| Dinov3          | dinov3_vitl16            | dinov3_vitl16_pretrain.pth       | ~300M                    |
| Dinov3          | dinov3_vith16plus        | dinov3_vith16plus_pretrain.pth   | ~840M                    |
| Dinov3          | dinov3_vit7b16           | dinov3_vit7b16_pretrain.pth      | ~6.7B                    |


**DINOv2**:

| MODEL           | MODEL_ARCH               | FINETUNE                     | SIZE                     |
|-----------------|--------------------------|------------------------------|--------------------------|
| Dinov2          | dinov2_vits14            | dinov2_vits14_pretrain.pth   | ~21M                     |
| Dinov2          | dinov2_vitb14            | dinov2_vitb14_pretrain.pth   | ~86M                     |
| Dinov2          | dinov2_vitl14            | dinov2_vitl14_pretrain.pth   | ~300M                    |
| Dinov2          | dinov2_vitg14            | dinov2_vitg14_pretrain.pth   | ~1.1B                    |


Change the DATA_PATH to your dataset directory.

```
# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"

# ==== Data settings ====
# change the dataset name and corresponding class number
DATASET="MESSIDOR2"
NUM_CLASS=5

# =======================
DATA_PATH="PATH TO THE DATASET"
TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --finetune "${FINETUNE}" \
  --savemodel \
  --global_pool \
  --batch_size 24 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "${TASK}" \
  --adaptation "${ADAPTATION}" 

```



6. For evaluation only (download data and model checkpoints [here](BENCHMARK.md); change the DATA_PATH below)


```
# ==== Model/settings (match training) ====
ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"

# ==== Data/settings (match training) ====
DATASET="MESSIDOR2"
NUM_CLASS=5

# =======================
DATA_PATH="PATH TO THE DATASET"
TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

# Path to the trained checkpoint (adjust if you saved elsewhere)
CKPT="./output_dir/${TASK}/checkpoint-best.pth"

# ==== Evaluation only ====
torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --savemodel \
  --global_pool \
  --batch_size 128 \
  --world_size 1 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "${TASK}" \
  --adaptation "${ADAPTATION}" \
  --eval \
  --resume "${CKPT}"

```

# Running scripts

### Run the pre-training of MerMED-FM

```
torchrun --nproc-per-node 8 main_mermed.py --arch vit_base --batch_size_per_gpu 128 --num_workers 10 --local_crops_number 10 --patch_size 16 --weight_decay 0.04 --weight_decay_end 0.4 --layer_decay 0.8 --lr 5e-5 --min_lr 1e-06 --print_freq 50 --ncrops 12 --global_crops_scale 0.2 1 --local_crops_scale 0.05 0.2 --epochs 100 --optimizer adamw --momentum_teacher 0.9995 --drop_path_rate 0.1 --use_bn_in_head true --out_dim 131072 --partition_size 16384 --student_temp 0.1 --teacher_temp 0.07 --warmup_teacher_temp_epochs 10 --warmup_teacher_temp 0.04 --teacher_temp 0.07 --clip_grad 1 --warmup_epochs 10 --data_path <path/to/MedFM> --pretrained_path <path/to/pre-trained/ckpt>
```

## Acknowledgments

We thanks the following projects for reference of creating MerMED-FM:

- [MaSSL](https://github.com/sthalles/MaSSL)
- [DINO](https://github.com/facebookresearch/dino)
- [RETFound](https://github.com/rmaphoh/RETFound)

## Citation

If you find MerMED-FM useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{zhou2025mermedfm,
  title        = {Multimodal, Multi-Disease Medical Imaging Foundation Model (MerMED-FM)},
  author       = {Yang Zhou and Chrystie Wan Ning Quek and Jun Zhou and Yan Wang and Yang Bai and Yuhe Ke and Jie Yao and Laura Gutierrez and Zhen Ling Teo and Darren Shu Jeng Ting and Brian T. Soetikno and Christopher S. Nielsen and Tobias Elze and Zengxiang Li and Linh Le Dinh and Lionel Tim-Ee Cheng and Tran Nguyen Tuan Anh and Chee Leong Cheng and Tien Yin Wong and Nan Liu and Iain Beehuat Tan and Tony Kiat Hon Lim and Rick Siow Mong Goh and Yong Liu and Daniel Shu Wei Ting},
  journal      = {arXiv preprint arXiv:2507.00185},
  year         = {2025},
  doi          = {10.48550/arXiv.2507.00185},
  url          = {https://arxiv.org/abs/2507.00185},
  archivePrefix= {arXiv},
  eprint       = {2507.00185},
  primaryClass = {eess.IV}
}
```
