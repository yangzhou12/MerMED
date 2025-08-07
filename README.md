# MerMED
PyTroch implementation for *Multimodal, Multi-Disease Medical Imaging Foundation Model (MerMED-FM)*

# Running scripts

### Run vit-base model

```
torchrun --nproc-per-node 8 main_mermed.py --arch vit_base --batch_size_per_gpu 128 --num_workers 10 --local_crops_number 10 --patch_size 16 --weight_decay 0.04 --weight_decay_end 0.4 --layer_decay 0.8 --lr 5e-5 --min_lr 1e-06 --print_freq 50 --global_crops_scale 0.2 1 --local_crops_scale 0.05 0.2 --epochs 400 --optimizer adamw --momentum_teacher 0.9995 --drop_path_rate 0.1 --use_bn_in_head true --out_dim 131072 --partition_size 16384 --warmup_teacher_temp_epochs 10 --warmup_teacher_temp 0.04 --teacher_temp 0.07 --clip_grad 1 --warmup_epochs 10 --data_path <path/to/MedFM> --pretrained_path <path/to/pre-trained/ckpt>
```