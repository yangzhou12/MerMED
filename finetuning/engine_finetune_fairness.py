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
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, average_precision_score, balanced_accuracy_score
from sklearn.calibration import calibration_curve
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np

def compute_ece_binary(y_true_binary, y_prob, n_bins=15):
    """Expected Calibration Error for binary labels and predicted probabilities.

    Args:
        y_true_binary (np.ndarray): shape (N,), values in {0,1}
        y_prob (np.ndarray): shape (N,), predicted probability for the positive class
        n_bins (int): number of bins

    Returns:
        float: ECE value in [0,1]
    """
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


def compute_ece_top1_multiclass(y_true_labels, y_prob_matrix, n_bins=15):
    """Top-1 ECE for multi-class classification using predicted confidence of argmax class.

    Args:
        y_true_labels (np.ndarray): shape (N,), true class indices
        y_prob_matrix (np.ndarray): shape (N, C), softmax probabilities per class
        n_bins (int): number of bins

    Returns:
        float: overall ECE
        np.ndarray: per-bin accuracies (for plotting)
        np.ndarray: per-bin confidences (for plotting)
        np.ndarray: per-bin counts
    """
    if y_prob_matrix.size == 0:
        return 0.0, np.array([]), np.array([]), np.array([])

    confidences = np.max(y_prob_matrix, axis=1)
    predictions = np.argmax(y_prob_matrix, axis=1)
    correctness = (predictions == y_true_labels).astype(np.float32)

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
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob, n_bins=15, strategy='uniform')
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

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    sensitivities = []
    specificities = []
    f1_scores = []
    precisions = []
    accuracies = []

    cm = confusion_matrix(y_true, y_pred)

    for i in range(len(cm)):
        class_acc = (cm[i, i] + np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]) / np.sum(cm)
        accuracies.append(class_acc)
        
        sensitivity = recall_score(y_true, y_pred, labels=[i], average='macro')
        sensitivities.append(sensitivity)
        
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp)
        specificities.append(specificity)
        
        precision = precision_score(y_true, y_pred, labels=[i], average='macro')
        precisions.append(precision)
        
        f1 = f1_score(y_true, y_pred, labels=[i], average='macro')
        f1_scores.append(f1)

    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precisions)
    avg_acc = np.mean(accuracies)

    return acc, avg_sensitivity, avg_specificity, avg_f1, sensitivities, specificities, f1_scores, precisions, accuracies


def compute_fairness_metrics(y_true, y_pred, y_prob, sensitive_attrs_dict, num_class):
    """
    Compute fairness metrics across sensitive attribute groups.
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        y_prob: Prediction probabilities (numpy array, shape: [N, num_classes])
        sensitive_attrs_dict: Dictionary of sensitive attributes {attr_name: numpy array of values}
        num_class: Number of classes
        
    Returns:
        Dictionary of fairness metrics
    """
    fairness_metrics = {}
    
    for attr_name, attr_values in sensitive_attrs_dict.items():
        # Get unique groups (excluding missing values marked as -1)
        unique_groups = np.unique(attr_values)
        # unique_groups = unique_groups[unique_groups != -1]
        unique_groups = unique_groups[np.isin(unique_groups, [0, 1])]
        
        if len(unique_groups) < 2:
            continue
            
        group_metrics = {}
        
        for group_val in unique_groups:
            mask = attr_values == group_val
            if np.sum(mask) < 10:  # Skip groups with too few samples
                continue
                
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            group_prob = y_prob[mask]
            
            # Compute metrics for this group
            try:
                acc = accuracy_score(group_true, group_pred)
                balanced_acc = balanced_accuracy_score(group_true, group_pred)
                
                # One-hot encode for AUC computation
                group_true_onehot = np.eye(num_class)[group_true]
                
                try:
                    auc_roc = roc_auc_score(group_true_onehot, group_prob, multi_class='ovr', average='macro')
                except ValueError:
                    auc_roc = -1
                    
                auc_pr = average_precision_score(group_true_onehot, group_prob, average='macro')
                f1 = f1_score(group_true, group_pred, average='macro', zero_division=0)
                
                # ECE for calibration fairness
                ece, _, _, _ = compute_ece_top1_multiclass(group_true, group_prob, n_bins=15)
                
                group_metrics[f'group_{group_val}'] = {
                    'count': int(np.sum(mask)),
                    'acc': float(acc),
                    'balanced_acc': float(balanced_acc),
                    'auc_roc': float(auc_roc),
                    'auc_pr': float(auc_pr),
                    'f1': float(f1),
                    'ece': float(ece)
                }
            except Exception as e:
                print(f"Warning: Could not compute metrics for {attr_name} group {group_val}: {e}")
                continue
        
        if len(group_metrics) >= 2:
            # Compute disparity metrics
            accs = [m['acc'] for m in group_metrics.values()]
            balanced_accs = [m['balanced_acc'] for m in group_metrics.values()]
            auc_rocs = [m['auc_roc'] for m in group_metrics.values() if m['auc_roc'] != -1]
            f1s = [m['f1'] for m in group_metrics.values()]
            eces = [m['ece'] for m in group_metrics.values()]
            
            # Demographic Parity Difference (max difference in positive prediction rate)
            pred_rates = []
            for group_val in unique_groups:
                mask = attr_values == group_val
                if np.sum(mask) >= 10:
                    # Positive prediction rate (proportion of positive predictions)
                    pred_rate = np.mean(y_pred[mask] > 0)  # For binary, or use specific class
                    pred_rates.append(pred_rate)
            
            fairness_metrics[attr_name] = {
                'group_metrics': group_metrics,
                'disparities': {
                    'acc_max_diff': float(np.max(accs) - np.min(accs)) if accs else 0.0,
                    'acc_std': float(np.std(accs)) if accs else 0.0,
                    'balanced_acc_max_diff': float(np.max(balanced_accs) - np.min(balanced_accs)) if balanced_accs else 0.0,
                    'balanced_acc_std': float(np.std(balanced_accs)) if balanced_accs else 0.0,
                    'auc_roc_max_diff': float(np.max(auc_rocs) - np.min(auc_rocs)) if auc_rocs else 0.0,
                    'auc_roc_std': float(np.std(auc_rocs)) if auc_rocs else 0.0,
                    'f1_max_diff': float(np.max(f1s) - np.min(f1s)) if f1s else 0.0,
                    'f1_std': float(np.std(f1s)) if f1s else 0.0,
                    'ece_max_diff': float(np.max(eces) - np.min(eces)) if eces else 0.0,
                    'ece_std': float(np.std(eces)) if eces else 0.0,
                    'demographic_parity_diff': float(np.max(pred_rates) - np.min(pred_rates)) if len(pred_rates) >= 2 else 0.0,
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
            
            if os.path.exists(group_path):
                group_df.to_csv(group_path, mode='w', header=True, index=False)
            else:
                group_df.to_csv(group_path, mode='w', header=True, index=False)
        
        # Save disparity metrics
        disparity_row = {'Epoch': epoch, 'Attribute': attr_name}
        disparity_row.update(attr_data['disparities'])
        disparity_df = pd.DataFrame([disparity_row])
        disparity_path = os.path.join(results_dir, f'fairness_{attr_name}_disparities_{mode}.csv')
        
        if os.path.exists(disparity_path):
            disparity_df.to_csv(disparity_path, mode='w', header=True, index=False)
        else:
            disparity_df.to_csv(disparity_path, mode='w', header=True, index=False)


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
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, results_dir, epoch, mode, num_class, 
             evaluate_fairness=False, sensitive_attr_names=None):
    """
    Evaluate model with optional fairness metrics.
    
    Args:
        data_loader: DataLoader
        model: Model to evaluate
        device: Device to use
        results_dir: Directory to save results
        epoch: Current epoch
        mode: 'val' or 'test'
        num_class: Number of classes
        evaluate_fairness: Whether to compute fairness metrics
        sensitive_attr_names: List of sensitive attribute names to evaluate
    """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    prediction_decode_list = []
    prediction_list = []
    logit_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # For fairness evaluation
    sensitive_attrs_lists = {} if evaluate_fairness else None
    if evaluate_fairness and sensitive_attr_names:
        for attr_name in sensitive_attr_names:
            sensitive_attrs_lists[attr_name] = []
    
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
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        with torch.amp.autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())
            logit_list.extend(output.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    true_label_array = np.array(true_label_onehot_list)
    prediction_array = np.array(prediction_list)
    logits_array = np.array(logit_list)

    # Convert sensitive attributes to numpy arrays
    if evaluate_fairness and sensitive_attrs_lists:
        for attr_name in sensitive_attrs_lists:
            sensitive_attrs_lists[attr_name] = np.array(sensitive_attrs_lists[attr_name])

    acc, sensitivity, specificity, F1, sensitivities, specificities, f1_scores, precisions, accuracies = compute_metrics(true_label_decode_list, prediction_decode_list)
    balanced_acc = balanced_accuracy_score(true_label_decode_list, prediction_decode_list)
    
    # Calculate per-class brier scores
    class_brier_scores = []
    for i in range(num_class):
        class_brier = np.mean((prediction_array[:, i] - true_label_array[:, i]) ** 2)
        class_brier_scores.append(class_brier)
    
    brier_score = np.mean(class_brier_scores)

    # Compute ECE
    ece_overall, bin_acc_overall, bin_conf_overall, _ = compute_ece_top1_multiclass(
        true_label_decode_list, prediction_array, n_bins=15
    )
    class_eces = []
    for i in range(num_class):
        class_ece = compute_ece_binary(true_label_array[:, i], prediction_array[:, i], n_bins=15)
        class_eces.append(class_ece)

    # Calculate per-class AUC-ROC and AUC-PR
    class_auc_rocs = []
    class_auc_prs = []
    for i in range(num_class):
        try:
            class_auc_roc = roc_auc_score(true_label_array[:, i], prediction_array[:, i])
        except ValueError:
            class_auc_roc = -1
        class_auc_pr = average_precision_score(true_label_array[:, i], prediction_array[:, i])
        class_auc_rocs.append(class_auc_roc)
        class_auc_prs.append(class_auc_pr)

    try:
        auc_roc = roc_auc_score(true_label_onehot_list, prediction_list, multi_class='ovr', average='macro')
    except ValueError:
        auc_roc = -1
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list, average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} BalancedAcc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f}'.format(acc, balanced_acc, auc_roc, auc_pr, F1)) 
    
    # Compute fairness metrics if requested
    fairness_metrics = None
    if evaluate_fairness and sensitive_attrs_lists and len(sensitive_attrs_lists) > 0:
        print("\nComputing fairness metrics...")
        fairness_metrics = compute_fairness_metrics(
            true_label_decode_list, 
            prediction_decode_list, 
            prediction_array,
            sensitive_attrs_lists,
            num_class
        )
        
        # Print fairness summary
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
    
    results_path = os.path.join(results_dir, 'metrics_{}.csv'.format(mode))
    
    # Create base results dictionary with overall metrics
    results = [{
        "Epoch": epoch,
        "Acc": acc,
        "BalancedAcc": balanced_acc,
        "AUCROC": auc_roc,
        "AUCPR": auc_pr,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Brier": brier_score,
        "F1": F1,
        "ECE_overall": ece_overall
    }]
    
    # Add per-class metrics
    for i in range(num_class):
        results[0].update({
            f"Class_{i}_Acc": accuracies[i],
            f"Class_{i}_AUCROC": class_auc_rocs[i],
            f"Class_{i}_AUCPR": class_auc_prs[i],
            f"Class_{i}_Sensitivity": sensitivities[i],
            f"Class_{i}_Specificity": specificities[i],
            f"Class_{i}_Brier": class_brier_scores[i],
            f"Class_{i}_F1": f1_scores[i],
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
    
    results = pd.DataFrame(results)

    if os.path.exists(results_path):
        results.to_csv(results_path, mode='w', header=True, index=False)
    else:
        results.to_csv(results_path, mode='w', header=True, index=False)
            
    # Save reliability diagrams
    try:
        os.makedirs(results_dir, exist_ok=True)
        overall_plot_path = os.path.join(results_dir, f'reliability_overall_{mode}.png')
        plot_reliability_diagram_overall(bin_conf_overall, bin_acc_overall, overall_plot_path,
                                         title=f'Reliability Diagram (Overall) - {mode}')
        for i in range(num_class):
            class_plot_path = os.path.join(results_dir, f'reliability_class_{i}_{mode}.png')
            plot_reliability_diagram_binary(true_label_array[:, i], prediction_array[:, i], class_plot_path,
                                            title_prefix=f'Class {i}')
    except Exception as _:
        pass

    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches ='tight')
        
        # Save full per-sample outputs for post-hoc metric recomputation
        try:
            # CSV with labels, predictions, per-class probabilities and logits
            data_dict = {
                'true_label': true_label_decode_list,
                'pred_label': prediction_decode_list,
            }
            for i in range(num_class):
                data_dict[f'prob_{i}'] = prediction_array[:, i]
            for i in range(num_class):
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
                'true_label': true_label_decode_list,
                'true_label_onehot': true_label_array,
                'pred_label': prediction_decode_list,
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
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc, auc_pr, F1


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
        
        # Color bars by magnitude (red for high disparity)
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