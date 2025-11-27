# MerMED-FM
PyTroch implementation for "[MerMED-FM: Multimodal, Multi-Disease Medical Imaging Foundation Model](https://arxiv.org/abs/2507.00185)"

# Running scripts

### Run vit-base model

```
torchrun --nproc-per-node 8 main_mermed.py --arch vit_base --batch_size_per_gpu 128 --num_workers 10 --local_crops_number 10 --patch_size 16 --weight_decay 0.04 --weight_decay_end 0.4 --layer_decay 0.8 --lr 5e-5 --min_lr 1e-06 --print_freq 50 --global_crops_scale 0.2 1 --local_crops_scale 0.05 0.2 --epochs 400 --optimizer adamw --momentum_teacher 0.9995 --drop_path_rate 0.1 --use_bn_in_head true --out_dim 131072 --partition_size 16384 --warmup_teacher_temp_epochs 10 --warmup_teacher_temp 0.04 --teacher_temp 0.07 --clip_grad 1 --warmup_epochs 10 --data_path <path/to/MedFM> --pretrained_path <path/to/pre-trained/ckpt>
```

## Acknowledgments

We thanks the following projects for reference of creating MerMED-FM:

- [MaSSL](https://github.com/sthalles/MaSSL)
- [DINO](https://github.com/facebookresearch/dino)

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
