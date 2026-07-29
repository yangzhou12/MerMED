# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import accuracy
from timm.data import Mixup
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix, average_precision_score, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_metrics(y_true, y_pred, y_prob=None, task_type='multi-class'):
    """Calculate comprehensive metrics for classification tasks"""
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if task_type == 'multi-class':
            cm = confusion_matrix(y_true, y_pred)
            n_classes = len(np.unique(y_true))
            
            # Per-class metrics
            for i in range(n_classes):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                tn = np.sum(cm) - (tp + fp + fn)
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                metrics[f'class_{i}_sensitivity'] = sensitivity
                metrics[f'class_{i}_specificity'] = specificity
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(
                    y_true, y_prob,
                    multi_class='ovr' if task_type == 'multi-class' else None,
                    average='macro'
                )
                metrics['au_pr'] = average_precision_score(y_true, y_prob, average='macro')
                metrics['brier_score'] = np.mean(np.sum((y_prob - y_true) ** 2, axis=1))
            except ValueError as e:
                print(f"Warning: Could not calculate some probability-based metrics: {e}")
                
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'accuracy': 0.0}
        
    return metrics

def compute_ece_binary(y_true_binary, y_prob, n_bins=15):
    """Expected Calibration Error for binary labels and predicted probabilities."""
    if y_prob.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_count = y_prob.shape[0]
    for b in range(n_bins):
        start = bin_edges[b]
        end = bin_edges[b + 1]
        mask = (y_prob > start) & (y_prob <= end) if b > 0 else (y_prob >= start) & (y_prob <= end)
        if not np.any(mask):
            continue
        conf = np.mean(y_prob[mask])
        acc = np.mean(y_true_binary[mask])
        weight = np.sum(mask) / total_count
        ece += weight * np.abs(acc - conf)
    return float(ece)


def compute_ece_multilabel(y_true, y_pred_binary, y_prob, n_bins=15):
    """
    Compute overall ECE for multilabel classification.
    Uses the max confidence across all classes for each sample.
    """
    if y_prob.size == 0:
        return 0.0, np.array([]), np.array([]), np.array([])
    
    # For multilabel, use max probability across all classes as confidence
    confidences = np.max(y_prob, axis=1).astype(np.float32)
    # Correctness is 1 if all predicted labels match true labels
    correctness = (y_pred_binary == y_true).all(axis=1).astype(np.float32)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_counts = []
    ece = 0.0
    total_count = confidences.shape[0]
    
    for b in range(n_bins):
        start = bin_edges[b]
        end = bin_edges[b + 1]
        mask = (confidences > start) & (confidences <= end) if b > 0 else (confidences >= start) & (confidences <= end)
        count_b = int(np.sum(mask))
        if count_b == 0:
            continue
        conf_b = float(np.mean(confidences[mask]))
        acc_b = float(np.mean(correctness[mask]))
        weight = count_b / total_count
        ece += weight * abs(acc_b - conf_b)
        bin_acc.append(acc_b)
        bin_conf.append(conf_b)
        bin_counts.append(count_b)
    
    return float(ece), np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)


def plot_reliability_diagram_overall(bin_conf, bin_acc, save_path, title="Reliability Diagram (Overall)"):
    """Plot overall reliability diagram using per-bin confidence and accuracy."""
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    if bin_conf.size > 0:
        plt.plot(bin_conf, bin_acc, marker='o', linewidth=2, label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reliability_diagram_binary(y_true_binary, y_prob, save_path, title_prefix="Class"):
    """Plot per-class reliability diagram using sklearn.calibration.calibration_curve."""
    if y_prob.size == 0:
        return
    try:
        prob_true, prob_pred = calibration_curve(y_true_binary.astype(np.float32), y_prob.astype(np.float32), n_bins=15, strategy='uniform')
    except Exception:
        prob_true, prob_pred = np.array([]), np.array([])
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    if prob_pred.size > 0:
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Empirical probability')
    plt.title(f"{title_prefix} Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_fairness_metrics_multilabel(y_true, y_pred, y_prob, sensitive_attrs_dict, num_classes):
    """
    Compute fairness metrics across sensitive attribute groups for multilabel classification.
    
    Args:
        y_true: True labels (numpy array, shape: [N, num_classes])
        y_pred: Predicted labels (numpy array, shape: [N, num_classes])
        y_prob: Prediction probabilities (numpy array, shape: [N, num_classes])
        sensitive_attrs_dict: Dictionary of sensitive attributes {attr_name: numpy array of values}
        num_classes: Number of classes
        
    Returns:
        Dictionary of fairness metrics
    """
    fairness_metrics = {}
    
    for attr_name, attr_values in sensitive_attrs_dict.items():
        # Get unique groups (excluding missing values marked as -1)
        unique_groups = np.unique(attr_values)
        unique_groups = unique_groups[np.isin(unique_groups, [0, 1])]
        
        if len(unique_groups) < 2:
            continue
            
        group_metrics = {}
        
        for group_val in unique_groups:
            mask = attr_values == group_val
            if np.sum(mask) < 10:  # Skip groups with too few samples
                continue
                
            group_true = y_true[mask].astype(np.float32)
            group_pred = y_pred[mask].astype(np.float32)
            group_prob = y_prob[mask].astype(np.float32)
            
            # Compute metrics for this group (macro-averaged across classes)
            try:
                acc = accuracy_score(group_true, group_pred)
                balanced_acc = np.mean([balanced_accuracy_score(group_true[:, i], group_pred[:, i]) 
                                       for i in range(num_classes)])
                
                # Per-class metrics for multilabel
                class_accs = []
                class_f1s = []
                class_auc_rocs = []
                class_auc_prs = []
                class_eces = []
                
                for i in range(num_classes):
                    class_true = group_true[:, i].astype(np.float32)
                    class_pred = group_pred[:, i].astype(np.float32)
                    class_prob = group_prob[:, i].astype(np.float32)
                    
                    class_acc = accuracy_score(class_true, class_pred)
                    class_accs.append(class_acc)
                    
                    try:
                        class_f1 = f1_score(class_true, class_pred, zero_division=0)
                        class_f1s.append(class_f1)
                    except:
                        class_f1s.append(0.0)
                    
                    try:
                        class_auc_roc = roc_auc_score(class_true, class_prob)
                        class_auc_rocs.append(class_auc_roc)
                    except ValueError:
                        class_auc_rocs.append(-1)
                    
                    try:
                        class_auc_pr = average_precision_score(class_true, class_prob)
                        class_auc_prs.append(class_auc_pr)
                    except:
                        class_auc_prs.append(0.0)
                    
                    class_ece = compute_ece_binary(class_true, class_prob, n_bins=15)
                    class_eces.append(class_ece)
                
                # Macro-averaged metrics
                macro_acc = np.mean(class_accs)
                macro_f1 = np.mean(class_f1s)
                macro_auc_roc = np.mean([x for x in class_auc_rocs if x != -1]) if any(x != -1 for x in class_auc_rocs) else -1
                macro_auc_pr = np.mean(class_auc_prs)
                macro_ece = np.mean(class_eces)
                
                # Overall multilabel metrics - ensure float32 for scipy
                overall_auc_roc = roc_auc_score(group_true.astype(np.float32), group_prob.astype(np.float32), average='macro') if group_true.size > 0 else -1
                overall_auc_pr = average_precision_score(group_true.astype(np.float32), group_prob.astype(np.float32), average='macro') if group_true.size > 0 else -1
                overall_f1 = f1_score(group_true, group_pred, average='macro', zero_division=0)
                
                group_metrics[f'group_{group_val}'] = {
                    'count': int(np.sum(mask)),
                    'acc': float(acc),
                    'balanced_acc': float(balanced_acc),
                    'macro_acc': float(macro_acc),
                    'macro_f1': float(macro_f1),
                    'macro_auc_roc': float(macro_auc_roc),
                    'macro_auc_pr': float(macro_auc_pr),
                    'macro_ece': float(macro_ece),
                    'overall_auc_roc': float(overall_auc_roc),
                    'overall_auc_pr': float(overall_auc_pr),
                    'overall_f1': float(overall_f1),
                    'f1': float(overall_f1),  # For compatibility with visualization
                    'auc_roc': float(overall_auc_roc) if overall_auc_roc != -1 else 0.0,  # For compatibility
                    'ece': float(macro_ece)  # For compatibility with visualization
                }
            except Exception as e:
                print(f"Warning: Could not compute metrics for {attr_name} group {group_val}: {e}")
                continue
        
        if len(group_metrics) >= 2:
            # Compute disparity metrics
            accs = [m['acc'] for m in group_metrics.values()]
            balanced_accs = [m['balanced_acc'] for m in group_metrics.values()]
            macro_accs = [m['macro_acc'] for m in group_metrics.values()]
            macro_f1s = [m['macro_f1'] for m in group_metrics.values()]
            macro_auc_rocs = [m['macro_auc_roc'] for m in group_metrics.values() if m['macro_auc_roc'] != -1]
            macro_auc_prs = [m['macro_auc_pr'] for m in group_metrics.values()]
            macro_eces = [m['macro_ece'] for m in group_metrics.values()]
            overall_auc_rocs = [m['overall_auc_roc'] for m in group_metrics.values() if m['overall_auc_roc'] != -1]
            overall_f1s = [m['overall_f1'] for m in group_metrics.values()]
            
            # Demographic Parity Difference (max difference in positive prediction rate per class)
            pred_rate_diffs = []
            for i in range(num_classes):
                pred_rates = []
                for group_val in unique_groups:
                    mask = attr_values == group_val
                    if np.sum(mask) >= 10:
                        pred_rate = np.mean(y_pred[mask, i])
                        pred_rates.append(pred_rate)
                if len(pred_rates) >= 2:
                    pred_rate_diffs.append(np.max(pred_rates) - np.min(pred_rates))
            
            fairness_metrics[attr_name] = {
                'group_metrics': group_metrics,
                'disparities': {
                    'acc_max_diff': float(np.max(accs) - np.min(accs)) if accs else 0.0,
                    'acc_std': float(np.std(accs)) if accs else 0.0,
                    'balanced_acc_max_diff': float(np.max(balanced_accs) - np.min(balanced_accs)) if balanced_accs else 0.0,
                    'balanced_acc_std': float(np.std(balanced_accs)) if balanced_accs else 0.0,
                    'macro_acc_max_diff': float(np.max(macro_accs) - np.min(macro_accs)) if macro_accs else 0.0,
                    'macro_f1_max_diff': float(np.max(macro_f1s) - np.min(macro_f1s)) if macro_f1s else 0.0,
                    'macro_auc_roc_max_diff': float(np.max(macro_auc_rocs) - np.min(macro_auc_rocs)) if macro_auc_rocs else 0.0,
                    'macro_auc_pr_max_diff': float(np.max(macro_auc_prs) - np.min(macro_auc_prs)) if macro_auc_prs else 0.0,
                    'macro_ece_max_diff': float(np.max(macro_eces) - np.min(macro_eces)) if macro_eces else 0.0,
                    'overall_auc_roc_max_diff': float(np.max(overall_auc_rocs) - np.min(overall_auc_rocs)) if overall_auc_rocs else 0.0,
                    'overall_f1_max_diff': float(np.max(overall_f1s) - np.min(overall_f1s)) if overall_f1s else 0.0,
                    'f1_max_diff': float(np.max(overall_f1s) - np.min(overall_f1s)) if overall_f1s else 0.0,  # For compatibility
                    'auc_roc_max_diff': float(np.max(overall_auc_rocs) - np.min(overall_auc_rocs)) if overall_auc_rocs else 0.0,  # For compatibility
                    'ece_max_diff': float(np.max(macro_eces) - np.min(macro_eces)) if macro_eces else 0.0,  # For compatibility
                    'demographic_parity_diff': float(np.mean(pred_rate_diffs)) if pred_rate_diffs else 0.0,
                }
            }
    
    return fairness_metrics


def save_fairness_results(fairness_metrics, results_dir, epoch, mode):
    """Save fairness metrics to CSV files."""
    for attr_name, attr_data in fairness_metrics.items():
        # Save group-level metrics
        group_data = []
        for group_name, metrics in attr_data['group_metrics'].items():
            row = {'Epoch': epoch, 'Attribute': attr_name, 'Group': group_name}
            row.update(metrics)
            group_data.append(row)
        
        if group_data:
            group_df = pd.DataFrame(group_data)
            group_path = os.path.join(results_dir, f'fairness_{attr_name}_groups_{mode}.csv')
            group_df.to_csv(group_path, mode='w', header=True, index=False)
        
        # Save disparity metrics
        disparity_row = {'Epoch': epoch, 'Attribute': attr_name}
        disparity_row.update(attr_data['disparities'])
        disparity_df = pd.DataFrame([disparity_row])
        disparity_path = os.path.join(results_dir, f'fairness_{attr_name}_disparities_{mode}.csv')
        disparity_df.to_csv(disparity_path, mode='w', header=True, index=False)


def plot_fairness_visualizations(fairness_metrics, results_dir, mode):
    """Generate visualization plots for fairness metrics."""
    for attr_name, attr_data in fairness_metrics.items():
        group_metrics = attr_data['group_metrics']
        
        if len(group_metrics) < 2:
            continue
        
        # Extract data for plotting
        groups = list(group_metrics.keys())
        groups_display = [g.replace('group_', '') for g in groups]
        
        metrics_to_plot = ['acc', 'balanced_acc', 'auc_roc', 'f1', 'ece']
        metric_labels = ['Accuracy', 'Balanced Accuracy', 'AUC-ROC', 'F1-Score', 'ECE']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[idx]
            values = [group_metrics[g][metric] for g in groups]
            
            # Bar plot
            bars = ax.bar(groups_display, values, alpha=0.7, edgecolor='black')
            
            # Color bars by value (red for low, green for high)
            if metric != 'ece':  # Higher is better
                colors = plt.cm.RdYlGn([v for v in values])
            else:  # Lower is better for ECE
                colors = plt.cm.RdYlGn_r([v for v in values])
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xlabel('Group')
            ax.set_ylabel(label)
            ax.set_title(f'{label} by {attr_name}')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Add sample counts in the last subplot
        ax = axes[5]
        counts = [group_metrics[g]['count'] for g in groups]
        bars = ax.bar(groups_display, counts, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Group')
        ax.set_ylabel('Sample Count')
        ax.set_title(f'Sample Distribution by {attr_name}')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'fairness_{attr_name}_{mode}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a disparity summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        disparities = attr_data['disparities']
        disparity_names = ['Acc', 'Bal. Acc', 'AUC-ROC', 'F1', 'ECE', 'Dem. Parity']
        disparity_values = [
            disparities['acc_max_diff'],
            disparities['balanced_acc_max_diff'],
            disparities['auc_roc_max_diff'],
            disparities['f1_max_diff'],
            disparities['ece_max_diff'],
            disparities['demographic_parity_diff']
        ]
        
        bars = ax.barh(disparity_names, disparity_values, alpha=0.7, edgecolor='black')
        
        # Color bars by magnitude
        colors = plt.cm.Reds([min(v * 5, 1.0) for v in disparity_values])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Disparity (Max Difference)')
        ax.set_title(f'Fairness Disparities by {attr_name} - {mode}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, disparity_values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{val:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        disparity_plot_path = os.path.join(results_dir, f'fairness_{attr_name}_disparities_{mode}.png')
        plt.savefig(disparity_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved fairness visualizations to {plot_path} and {disparity_plot_path}")

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Handle both with and without sensitive attributes
        if len(batch) == 3:
            samples, targets, _ = batch  # Ignore sensitive attrs during training
        else:
            samples, targets = batch

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, results_dir, epoch, mode, num_class, task_type='multi-class'):
    """
    Unified evaluation function for both multi-class and multi-label classification
    Args:
        data_loader: DataLoader object
        model: PyTorch model
        device: torch device
        results_dir: directory to save results
        epoch: current epoch
        mode: 'train', 'val', or 'test'
        num_class: number of classes
        task_type: 'multi-class' or 'multi-label'
    """
    if not data_loader:
        raise ValueError("Empty data loader provided")
    
    if task_type not in ['multi-class', 'multi-label']:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'multi-class' or 'multi-label'")

    criterion = torch.nn.CrossEntropyLoss() if task_type == 'multi-class' else torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    prediction_decode_list = []
    prediction_list = []
    true_label_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(images)
            loss = criterion(output, target)

            if task_type == 'multi-class':
                prediction_softmax = nn.Softmax(dim=1)(output)
                _,prediction_decode = torch.max(prediction_softmax, 1)
                true_label = F.one_hot(target.to(torch.int64), num_classes=num_class)
                _,true_label_decode = torch.max(true_label, 1)

                prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
                true_label_list.extend(true_label_decode.cpu().detach().numpy())
                prediction_list.extend(prediction_softmax.cpu().detach().numpy())
                true_label_onehot_list = true_label.cpu().detach().numpy()

                acc1, _ = accuracy(output, target, topk=(1,2))
            else:  # multi-label
                prediction_sigmoid = torch.sigmoid(output)
                prediction_decode = (prediction_sigmoid > 0.5).float()
                
                prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
                true_label_list.extend(target.cpu().detach().numpy())
                prediction_list.extend(prediction_sigmoid.cpu().detach().numpy())

                acc1 = multilabel_accuracy(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # Convert lists to numpy arrays
    true_label_array = np.array(true_label_list)
    prediction_array = np.array(prediction_list)
    prediction_decode_array = np.array(prediction_decode_list)

    if task_type == 'multi-class':
        # Multi-class metrics
        metrics = calculate_metrics(true_label_array, prediction_decode_array, prediction_array, task_type='multi-class')
        acc = metrics['accuracy']
        sensitivity = np.mean([metrics[f'class_{i}_sensitivity'] for i in range(num_class)])
        specificity = np.mean([metrics[f'class_{i}_specificity'] for i in range(num_class)])
        F1 = f1_score(true_label_array, prediction_decode_array, average='macro')
        auc_roc = metrics['auc_roc']
        auc_pr = metrics['au_pr']
        brier_score = metrics['brier_score']

        print(f'Multi-class Metrics - Acc: {acc:.4f} AUC-roc: {auc_roc:.4f} AUC-pr: {auc_pr:.4f} F1-score: {F1:.4f}')
        results = [{"Epoch": epoch, "Acc": acc, "AUCROC": auc_roc, "AUCPR": auc_pr, 
                   "Sensitivity": sensitivity, "Specificity": specificity, 
                   "Brier": brier_score, "F1": F1}]
    else:
        # Multi-label metrics
        acc = accuracy_score(true_label_array, prediction_decode_array)
        f1 = f1_score(true_label_array, prediction_decode_array, average='macro')
        auc_roc = roc_auc_score(true_label_array, prediction_array, average='macro')
        auc_pr = average_precision_score(true_label_array, prediction_array, average='macro')
        brier_score = np.mean(np.sum((prediction_array - true_label_array) ** 2, axis=1))

        print(f'Multi-label Metrics - Acc: {acc:.4f} AUC-roc: {auc_roc:.4f} AUC-pr: {auc_pr:.4f} F1-score: {f1:.4f}')
        results = [{"Epoch": epoch, "Acc": acc, "AUCROC": auc_roc, "AUCPR": auc_pr, 
                   "F1": f1, "Brier": brier_score}]

    # Save results
    results_path = os.path.join(results_dir, f'metrics_{mode}.csv')
    results = pd.DataFrame(results)
    if os.path.exists(results_path):
        results.to_csv(results_path, mode='a', header=False, index=False)
    else:
        results.to_csv(results_path, mode='w', header=True, index=False)

    # Plot confusion matrix for multi-class test mode
    if mode == 'test' and task_type == 'multi-class':
        cm = ConfusionMatrix(actual_vector=true_label_array, predict_vector=prediction_decode_array)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    
    # Clear memory
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc

def multilabel_accuracy(output, target):
    """
    Compute the multi-label classification accuracy.
    """
    with torch.no_grad():
        output_binary = (torch.sigmoid(output) > 0.5).float()
        correct = (output_binary == target).float().sum(dim=1)
        total = target.size(0) * target.size(1)
        acc = correct.sum() / total
    return acc



@torch.no_grad()
def multilabel_evaluate(data_loader, model, device, results_dir, epoch, mode, num_classes,
                       evaluate_fairness=False, sensitive_attr_names=None):
    """
    Evaluate multilabel model with optional fairness metrics.
    
    Args:
        data_loader: DataLoader
        model: Model to evaluate
        device: Device to use
        results_dir: Directory to save results
        epoch: Current epoch
        mode: 'val' or 'test'
        num_classes: Number of classes
        evaluate_fairness: Whether to compute fairness metrics
        sensitive_attr_names: List of sensitive attribute names to evaluate
    """
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    prediction_decode_list = []
    true_label_list = []
    prediction_list = []
    logit_list = []
    
    # For fairness evaluation
    sensitive_attrs_lists = {} if evaluate_fairness else None
    if evaluate_fairness and sensitive_attr_names:
        for attr_name in sensitive_attr_names:
            sensitive_attrs_lists[attr_name] = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Handle both with and without sensitive attributes
        if len(batch) == 3:
            images, target, sensitive_attrs = batch
            # Store sensitive attributes for fairness evaluation
            if evaluate_fairness and sensitive_attrs_lists is not None:
                for attr_name in sensitive_attr_names:
                    if attr_name in sensitive_attrs:
                        sensitive_attrs_lists[attr_name].extend(
                            sensitive_attrs[attr_name].cpu().numpy()
                        )
        else:
            images, target = batch
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            prediction_sigmoid = torch.sigmoid(output)

            true_label_list.extend(target.cpu().detach().numpy())
            prediction_list.extend(prediction_sigmoid.cpu().detach().numpy())
            logit_list.extend(output.cpu().detach().numpy())

        acc1 = multilabel_accuracy(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # Convert lists to numpy arrays and ensure float32 dtype (not float16) for scipy compatibility
    true_label_array = np.array(true_label_list, dtype=np.float32)
    prediction_array = np.array(prediction_list, dtype=np.float32)
    logits_array = np.array(logit_list, dtype=np.float32)
    
    # Convert sensitive attributes to numpy arrays
    if evaluate_fairness and sensitive_attrs_lists:
        for attr_name in sensitive_attrs_lists:
            sensitive_attrs_lists[attr_name] = np.array(sensitive_attrs_lists[attr_name], dtype=np.int64)
    
    # Compute binary predictions using optimal threshold per class
    prediction_decode_array = np.zeros_like(prediction_array)

    # Initialize lists to store metrics for each class
    accuracies = []
    f1_scores = []
    auc_roc_scores = []
    auc_pr_scores = []
    sensitivities = []
    specificities = []
    per_class_metrics = {}
    for i in range(num_classes):
        gt_np = true_label_array[:, i].astype(np.float32)
        pred_np = prediction_array[:, i].astype(np.float32)

        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores_curve = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores_curve)
        max_f1_idx = np.argmax(f1_scores_curve)
        
        # Handle case where max_f1_idx might be out of bounds for thresholds
        if len(thresholds) > max_f1_idx:
            max_f1_thresh = thresholds[max_f1_idx]
        else:
            max_f1_thresh = 0.0

        # Calculate sensitivity (recall) at best F1
        best_sensitivity = recall[max_f1_idx]

        # Calculate specificity at best F1
        y_pred_binary = (pred_np >= max_f1_thresh).astype(int)
        prediction_decode_array[:, i] = y_pred_binary  # Store binary predictions
        tn = np.sum((gt_np == 0) & (y_pred_binary == 0))
        fp = np.sum((gt_np == 0) & (y_pred_binary == 1))
        best_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Store metrics for this class
        accuracies.append(accuracy_score(gt_np, y_pred_binary))
        f1_scores.append(max_f1)
        auc_roc_scores.append(roc_auc_score(gt_np, pred_np))
        auc_pr_scores.append(average_precision_score(gt_np, pred_np))
        sensitivities.append(best_sensitivity)
        specificities.append(best_specificity)

        class_metrics = {
            'accuracy': accuracies[-1],
            'f1': f1_scores[-1],
            'auc_roc': auc_roc_scores[-1],
            'auc_pr': auc_pr_scores[-1],
            'sensitivity': sensitivities[-1],
            'specificity': specificities[-1]
        }
        per_class_metrics[f'class_{i}'] = class_metrics

        # # Print per-class metrics
        # print(f'\nClass {i} Metrics:')
        # print(f'Accuracy: {class_metrics["accuracy"]:.4f}')
        # print(f'F1-score: {class_metrics["f1"]:.4f}')
        # print(f'AUC-ROC: {class_metrics["auc_roc"]:.4f}')
        # print(f'AUC-PR: {class_metrics["auc_pr"]:.4f}')
        # print(f'Sensitivity: {class_metrics["sensitivity"]:.4f}')
        # print(f'Specificity: {class_metrics["specificity"]:.4f}')

    # Compute overall multilabel metrics
    overall_acc = accuracy_score(true_label_array, prediction_decode_array)
    overall_f1 = f1_score(true_label_array, prediction_decode_array, average='macro', zero_division=0)
    try:
        overall_auc_roc = roc_auc_score(true_label_array, prediction_array, average='macro')
    except ValueError:
        overall_auc_roc = -1
    overall_auc_pr = average_precision_score(true_label_array, prediction_array, average='macro')
    
    # Compute balanced accuracy (macro-averaged across classes)
    balanced_acc = np.mean([balanced_accuracy_score(true_label_array[:, i], prediction_decode_array[:, i]) 
                           for i in range(num_classes)])
    
    # Compute macro-averaged metrics (per-class averages)
    macro_accuracy = np.mean(accuracies)
    macro_f1 = np.mean(f1_scores)
    macro_auc_roc = np.mean(auc_roc_scores)
    macro_auc_pr = np.mean(auc_pr_scores)
    macro_sensitivity = np.mean(sensitivities)
    macro_specificity = np.mean(specificities)
    
    # Calculate per-class brier scores
    class_brier_scores = []
    for i in range(num_classes):
        class_brier = np.mean((prediction_array[:, i] - true_label_array[:, i]) ** 2)
        class_brier_scores.append(class_brier)
    brier_score = np.mean(class_brier_scores)
    
    # Compute ECE
    ece_overall, bin_acc_overall, bin_conf_overall, _ = compute_ece_multilabel(
        true_label_array, prediction_decode_array, prediction_array, n_bins=15
    )
    class_eces = []
    for i in range(num_classes):
        class_ece = compute_ece_binary(true_label_array[:, i], prediction_array[:, i], n_bins=15)
        class_eces.append(class_ece)

    # Print metrics in format consistent with fairness engine
    print('Sklearn Metrics - Acc: {:.4f} BalancedAcc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f}'.format(
        overall_acc, balanced_acc, overall_auc_roc if overall_auc_roc != -1 else 0.0, overall_auc_pr, overall_f1))
    
    # Compute fairness metrics if requested
    fairness_metrics = None
    if evaluate_fairness and sensitive_attrs_lists and len(sensitive_attrs_lists) > 0:
        print("\nComputing fairness metrics...")
        fairness_metrics = compute_fairness_metrics_multilabel(
            true_label_array, 
            prediction_decode_array, 
            prediction_array,
            sensitive_attrs_lists,
            num_classes
        )
        
        # Print fairness summary (consistent with fairness engine format)
        for attr_name, attr_data in fairness_metrics.items():
            print(f"\nFairness metrics for {attr_name}:")
            print(f"  Number of groups: {len(attr_data['group_metrics'])}")
            disparities = attr_data['disparities']
            print(f"  Accuracy disparity (max diff): {disparities['acc_max_diff']:.4f}")
            print(f"  Balanced Accuracy disparity (max diff): {disparities['balanced_acc_max_diff']:.4f}")
            print(f"  AUC-ROC disparity (max diff): {disparities['auc_roc_max_diff']:.4f}")
            print(f"  F1 disparity (max diff): {disparities['f1_max_diff']:.4f}")
            print(f"  ECE disparity (max diff): {disparities['ece_max_diff']:.4f}")
            print(f"  Demographic parity difference: {disparities['demographic_parity_diff']:.4f}")
        
        # Save fairness results
        save_fairness_results(fairness_metrics, results_dir, epoch, mode)

    # Save results to CSV (consistent with fairness engine naming)
    results_path = os.path.join(results_dir, 'metrics_{}.csv'.format(mode))

    # Create base results dictionary with overall metrics (consistent with fairness engine)
    results = [{
        "Epoch": epoch,
        "Acc": overall_acc,
        "BalancedAcc": balanced_acc,
        "AUCROC": overall_auc_roc if overall_auc_roc != -1 else -1,
        "AUCPR": overall_auc_pr,
        "Sensitivity": macro_sensitivity,
        "Specificity": macro_specificity,
        "Brier": brier_score,
        "F1": overall_f1,
        "ECE_overall": ece_overall
    }]
    
    # Add per-class metrics (consistent with fairness engine naming)
    for i in range(num_classes):
        class_metrics = per_class_metrics[f'class_{i}']
        results[0].update({
            f"Class_{i}_Acc": class_metrics['accuracy'],
            f"Class_{i}_AUCROC": class_metrics['auc_roc'],
            f"Class_{i}_AUCPR": class_metrics['auc_pr'],
            f"Class_{i}_Sensitivity": class_metrics['sensitivity'],
            f"Class_{i}_Specificity": class_metrics['specificity'],
            f"Class_{i}_Brier": class_brier_scores[i],
            f"Class_{i}_F1": class_metrics['f1'],
            f"Class_{i}_ECE": class_eces[i]
        })
    
    # Add fairness summary to main results
    if fairness_metrics:
        for attr_name, attr_data in fairness_metrics.items():
            disparities = attr_data['disparities']
            results[0].update({
                f"{attr_name}_acc_disparity": disparities['acc_max_diff'],
                f"{attr_name}_balanced_acc_disparity": disparities['balanced_acc_max_diff'],
                f"{attr_name}_auc_roc_disparity": disparities['auc_roc_max_diff'],
                f"{attr_name}_f1_disparity": disparities['f1_max_diff'],
                f"{attr_name}_ece_disparity": disparities['ece_max_diff'],
                f"{attr_name}_demographic_parity": disparities['demographic_parity_diff']
            })

    # Convert to DataFrame and save
    results = pd.DataFrame(results)
    if os.path.exists(results_path):
        results.to_csv(results_path, mode='w', header=True, index=False)
    else:
        results.to_csv(results_path, mode='w', header=True, index=False)
    
    # Save reliability diagrams (consistent with fairness engine)
    try:
        os.makedirs(results_dir, exist_ok=True)
        overall_plot_path = os.path.join(results_dir, f'reliability_overall_{mode}.png')
        plot_reliability_diagram_overall(bin_conf_overall, bin_acc_overall, overall_plot_path,
                                         title=f'Reliability Diagram (Overall) - {mode}')
        for i in range(num_classes):
            class_plot_path = os.path.join(results_dir, f'reliability_class_{i}_{mode}.png')
            plot_reliability_diagram_binary(true_label_array[:, i], prediction_array[:, i], class_plot_path,
                                            title_prefix=f'Class {i}')
    except Exception as _:
        pass
    
    # Save full per-sample outputs for post-hoc metric recomputation (test mode only)
    if mode == 'test':
        try:
            # CSV with labels, predictions, per-class probabilities and logits
            data_dict = {}
            for i in range(num_classes):
                data_dict[f'true_label_{i}'] = true_label_array[:, i]
                data_dict[f'pred_label_{i}'] = prediction_decode_array[:, i]
                data_dict[f'prob_{i}'] = prediction_array[:, i]
                data_dict[f'logit_{i}'] = logits_array[:, i]
            
            # Add sensitive attributes to output CSV
            if evaluate_fairness and sensitive_attrs_lists:
                for attr_name, attr_values in sensitive_attrs_lists.items():
                    data_dict[f'sensitive_{attr_name}'] = attr_values
            
            outputs_df = pd.DataFrame(data_dict)
            outputs_csv_path = os.path.join(results_dir, f'outputs_{mode}.csv')
            outputs_df.to_csv(outputs_csv_path, index=False)

            # NPZ with full arrays
            outputs_npz_path = os.path.join(results_dir, f'outputs_{mode}.npz')
            npz_dict = {
                'true_label': true_label_array,
                'pred_label': prediction_decode_array,
                'probs': prediction_array,
                'logits': logits_array,
            }
            
            # Add sensitive attributes to NPZ
            if evaluate_fairness and sensitive_attrs_lists:
                for attr_name, attr_values in sensitive_attrs_lists.items():
                    npz_dict[f'sensitive_{attr_name}'] = attr_values
            
            np.savez_compressed(outputs_npz_path, **npz_dict)
        except Exception as e:
            print(f"Warning: Could not save outputs: {e}")
            pass
        
        # Generate fairness visualization plots
        if fairness_metrics:
            try:
                plot_fairness_visualizations(fairness_metrics, results_dir, mode)
            except Exception as e:
                print(f"Warning: Could not generate fairness plots: {e}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, overall_auc_roc if overall_auc_roc != -1 else 0.0

