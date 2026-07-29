"""Backbone builder selected by ``--model``.

MerMED-FM is a ViT-B/16, built by ``vit_base_patch16``. Weights are loaded from the
released checkpoint by the ``--finetune`` handling in the ``main_*`` entry points.
"""
from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
