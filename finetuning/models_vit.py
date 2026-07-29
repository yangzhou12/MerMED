"""Backbone builders selected by ``--model``.

``vit_base_patch16`` is MerMED-FM itself. The remaining builders exist to load
the comparison foundation models reported in the paper (BiomedCLIP, UniMed-CLIP,
UNI, DINOv2, Swin) through the same finetuning pipeline.
"""
from functools import partial

import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer, _create_swin_transformer


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def bmc_clip_cf(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        pre_norm=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def unimed_vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        pre_norm=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model = VisionTransformer(
    #     patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    #     pre_norm=True, init_values=1e-5, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def clip_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        pre_norm=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def dinov2_base_patch14(**kwargs):
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False)
    return model

def uni_vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, 
        init_values=1e-5, **kwargs)
    return model


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=3, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def swin_base_patch4_window7_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-B @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer(
        'swin_base_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))

def swin_large_patch4_window7_224(pretrained=False, **kwargs) -> SwinTransformer:
    """ Swin-L @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_transformer(
        'swin_large_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))
