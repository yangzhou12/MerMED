# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
"""Finetune the MerMED-FM ViT backbone with subgroup fairness evaluation.

Same as ``main_finetune.py`` but reports per-group metrics and fairness disparities
(e.g. across age/gender) using the fairness-aware dataset loader and engine.
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer

from util.datasets_fairness import build_dataset
from util.pos_embed import interpolate_pos_embed
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_finetune_fairness import train_one_epoch, evaluate
import models_vit
import re

def get_args_parser():
    parser = argparse.ArgumentParser('MerMED-FM fine-tuning for image classification')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.2)')

    
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--use_amp', action='store_true')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', default='avg', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path (for local datasets) or ignored (for Hugging Face datasets)')
    parser.add_argument('--label_path', default='', type=str,
                        help='path to labels.csv for local datasets, or Hugging Face dataset name for HF datasets')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--train_size', type=float, default=1.0, 
                        help='Percentage of training size')
    parser.add_argument('--modality', type=str, default=None, choices=['cfp', 'oct', 'cxr', 'pathology', 'CT', 'US', 'skin', 'other'],
                        help='Image modality type for proper normalization')
    
    # Fairness parameters
    parser.add_argument('--sensitive_attr', type=str, nargs='+', default=None,
                        help='Sensitive attributes for fairness evaluation (e.g., gender age race)')
    parser.add_argument('--evaluate_fairness', action='store_true',
                        help='Enable fairness evaluation during validation and testing')
    parser.add_argument('--fairness_eval_frequency', type=int, default=5,
                        help='Evaluate fairness every N epochs (default: 5, set to 1 for every epoch)')
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False

    # Set return_sensitive_attr based on whether fairness evaluation is enabled
    args.return_sensitive_attr = args.evaluate_fairness and args.sensitive_attr is not None
    
    # Print fairness configuration
    if args.return_sensitive_attr:
        print(f"\n{'='*60}")
        print("Fairness Evaluation Enabled")
        print(f"{'='*60}")
        print(f"Sensitive attributes: {args.sensitive_attr}")
        print(f"Evaluation frequency: Every {args.fairness_eval_frequency} epoch(s)")
        print(f"{'='*60}\n")

    dataset_train = build_dataset(data_split='train', args=args, train_size=args.train_size)
    dataset_val = build_dataset(data_split='val', args=args)
    dataset_test = build_dataset(data_split='test', args=args)
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")
    print(f"Test samples: {len(dataset_test)}")
    
    # Print sensitive attribute information if available
    if args.return_sensitive_attr and hasattr(dataset_train, 'get_sensitive_attr_info'):
        sens_info = dataset_train.get_sensitive_attr_info()
        print(f"\nSensitive Attribute Mappings:")
        for attr in sens_info['attributes']:
            mapping = sens_info['mappings'][attr]
            if mapping:
                print(f"  {attr}: {mapping}")
            else:
                print(f"  {attr}: numerical (no mapping)")
        
        # Print distribution
        print(f"\nSensitive Attribute Distribution:")
        for split_name, dataset in [("Train", dataset_train), ("Val", dataset_val), ("Test", dataset_test)]:
            if hasattr(dataset, 'get_sensitive_attr_distribution'):
                dist = dataset.get_sensitive_attr_distribution()
                print(f"\n{split_name} split:")
                for attr, values in dist.items():
                    print(f"  {attr}: {values}")

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        log_dir = os.path.join(args.log_dir, args.task)
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # MerMED is a ViT-B/16 backbone (--model vit_base_patch16).
    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        dynamic_img_size=True,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
            checkpoint_model = {k.replace("encoder.", ""): v for k, v in checkpoint_model.items()}
        elif 'state_dict' in checkpoint.keys():
            checkpoint_model = checkpoint['state_dict']
        elif 'student' in checkpoint.keys():
            checkpoint_model = checkpoint['student']
            prefix_mappings = {
                "module.backbone.encoder.": "",
                "module.backbone.": "",
                "backbone.": "",
            }
            
            scale_mappings = {
                ".gamma_1": ".ls1.gamma",
                ".gamma_2": ".ls2.gamma",
            }

            new_checkpoint_model = {}
            for old_key in checkpoint_model.keys():
                new_key = old_key
                for old_prefix, new_prefix in prefix_mappings.items():
                    if old_key.startswith(old_prefix):
                        new_key = old_key.replace(old_prefix, new_prefix)
                        break
                for old_prefix, new_prefix in scale_mappings.items():
                    if old_prefix in new_key:
                        new_key = new_key.replace(old_prefix, new_prefix)
                        break
                new_checkpoint_model[new_key] = checkpoint_model[old_key]
            checkpoint_model = new_checkpoint_model
        elif 'teacher' in checkpoint.keys():
            checkpoint_model = checkpoint['teacher']
            new_checkpoint_model = {}
            for old_key in checkpoint_model.keys():
                if "module.backbone." in old_key:
                    new_key = old_key.replace("module.backbone.", "")
                    new_checkpoint_model[new_key] = checkpoint_model[old_key]
                elif "backbone." in old_key:
                    new_key = old_key.replace("backbone.", "")
                    new_checkpoint_model[new_key] = checkpoint_model[old_key]
                else:
                    new_checkpoint_model[old_key] = checkpoint_model[old_key]

            def revert_block_chunk_weight(state_dict):
                return {(re.sub(r'blocks\.(\d+)\.(\d+)\.', r'blocks.\2.', k) if re.match(r'blocks\.\d+\.\d+\.', k) else k): v for k, v in state_dict.items()}
            
            new_checkpoint_model = revert_block_chunk_weight(new_checkpoint_model)
            checkpoint_model = new_checkpoint_model
        else:
            checkpoint_model = checkpoint

        for k in ['head.weight', 'head.bias', 'head.fc.weight', 'head.fc.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        for k in ['norm.weight', 'norm.bias', 'fc_norm.weight', 'fc_norm.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if hasattr(model, 'head') and hasattr(model.head, 'weight'):
            trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    args.opt = "adamw"
    args.momentum = 0.9
    optimizer = create_optimizer(args, model_without_ddp)

    loss_scaler = NativeScaler(args)

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.eval:
        print("\n" + "="*60)
        print("EVALUATION MODE - Testing with Fairness Analysis")
        print("="*60 + "\n")
        
        test_stats, auc_roc, auc_pr, F1 = evaluate(
            data_loader_test, model, device, output_dir, 
            epoch=0, mode='test', num_class=args.nb_classes,
            evaluate_fairness=args.evaluate_fairness,
            sensitive_attr_names=args.sensitive_attr
        )
        
        print("\n" + "="*60)
        print("Evaluation Complete")
        print("="*60)
        print(f"Overall Accuracy: {test_stats['acc1']:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"F1-Score: {F1:.4f}")
        print("="*60 + "\n")
        
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    max_F1 = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        # Determine if fairness should be evaluated this epoch
        evaluate_fairness_this_epoch = (
            args.evaluate_fairness and 
            args.sensitive_attr is not None and
            (epoch % args.fairness_eval_frequency == 0 or epoch == args.epochs - 1)
        )

        val_stats, val_auc_roc, val_auc_pr, val_F1 = evaluate(
            data_loader_val, model, device, output_dir,
            epoch, mode='val', num_class=args.nb_classes,
            evaluate_fairness=evaluate_fairness_this_epoch,
            sensitive_attr_names=args.sensitive_attr if evaluate_fairness_this_epoch else None
        )
        
        if max_auc < val_auc_roc:
            max_auc = val_auc_roc

            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        
        # Always evaluate fairness on test set at the last epoch
        if epoch == (args.epochs - 1):
            print("\n" + "="*60)
            print("FINAL EVALUATION - Testing with Fairness Analysis")
            print("="*60 + "\n")
            
            test_stats, auc_roc, auc_pr, F1 = evaluate(
                data_loader_test, model, device, output_dir,
                epoch, mode='test', num_class=args.nb_classes,
                evaluate_fairness=args.evaluate_fairness,
                sensitive_attr_names=args.sensitive_attr
            )
            
            print("\n" + "="*60)
            print("Final Test Results")
            print("="*60)
            print(f"Test Accuracy: {test_stats['acc1']:.4f}")
            print(f"Test AUC-ROC: {auc_roc:.4f}")
            print(f"Test AUC-PR: {auc_pr:.4f}")
            print(f"Test F1-Score: {F1:.4f}")
            print("="*60 + "\n")
        
        if log_writer is not None:
            log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
            log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)