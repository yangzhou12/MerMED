# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
# import csv
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
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, average_precision_score, balanced_accuracy_score
from sklearn.calibration import calibration_curve
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np

# import calibration as cal

def compute_ece_binary(y_true_binary, y_prob, n_bins=15):
    """Expected Calibration Error for binary labels and predicted probabilities.

    Args:
        y_true_binary (np.ndarray): shape (N,), values in {0,1}
        y_prob (np.ndarray): shape (N,), predicted probability for the positive class
        n_bins (int): number of bins

    Returns:
        float: ECE value in [0,1]
    """
    # Avoid empty input
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
    # Perfectly calibrated line
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

    # Initialize lists to store metrics for each class
    sensitivities = []
    specificities = []
    f1_scores = []
    precisions = []
    accuracies = []

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Loop over each class
    for i in range(len(cm)):
        # Class accuracy
        class_acc = (cm[i, i] + np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]) / np.sum(cm)
        accuracies.append(class_acc)
        
        # Sensitivity (Recall)
        sensitivity = recall_score(y_true, y_pred, labels=[i], average='macro')
        sensitivities.append(sensitivity)
        
        # Specificity
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp)
        specificities.append(specificity)
        
        # Precision
        precision = precision_score(y_true, y_pred, labels=[i], average='macro')
        precisions.append(precision)
        
        # F1 Score
        f1 = f1_score(y_true, y_pred, labels=[i], average='macro')
        f1_scores.append(f1)

    # Average metrics
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precisions)
    avg_acc = np.mean(accuracies)

    return acc, avg_sensitivity, avg_specificity, avg_f1, sensitivities, specificities, f1_scores, precisions, accuracies


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

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

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
def evaluate(data_loader, model, device, results_dir, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    prediction_decode_list = []
    prediction_list = []
    logit_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
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

    acc, sensitivity, specificity, F1, sensitivities, specificities, f1_scores, precisions, accuracies = compute_metrics(true_label_decode_list, prediction_decode_list)
    # Balanced accuracy accounts for class imbalance (macro recall)
    balanced_acc = balanced_accuracy_score(true_label_decode_list, prediction_decode_list)
    
    # Calculate per-class brier scores
    class_brier_scores = []
    for i in range(num_class):
        class_brier = np.mean((prediction_array[:, i] - true_label_array[:, i]) ** 2)
        class_brier_scores.append(class_brier)

    brier_score = np.mean(class_brier_scores)

    # Compute ECE: overall (top-1) and per-class (one-vs-rest)
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
    
    results = pd.DataFrame(results)

    if os.path.exists(results_path):
        # results.to_csv(results_path, mode='a', header=False, index=False)
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
            outputs_df = pd.DataFrame(data_dict)
            outputs_csv_path = os.path.join(results_dir, f'outputs_{mode}.csv')
            outputs_df.to_csv(outputs_csv_path, index=False)

            # NPZ with full arrays
            outputs_npz_path = os.path.join(results_dir, f'outputs_{mode}.npz')
            np.savez_compressed(
                outputs_npz_path,
                true_label=true_label_decode_list,
                true_label_onehot=true_label_array,
                pred_label=prediction_decode_list,
                probs=prediction_array,
                logits=logits_array,
            )
        except Exception:
            pass
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc, auc_pr, F1

