"""Cross-entropy criterion between student and teacher logits for MerMED.

Masks the -inf entries the memory bank introduces when it gates logits by label
agreement, so those positions contribute nothing to the loss.

Adapted from MaSSL (https://github.com/sthalles/MaSSL).
"""
import torch.nn as nn
import torch


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def my_cross_entropy(self, p, q, epsilon=1e-8):
        """
        Cross-entropy loss with masking for invalid values (e.g., -inf) directly during the summation step.

        Args:
            p (torch.Tensor): Logits for predictions.
            q (torch.Tensor): Logits or probabilities for targets.
            epsilon (float): Small value to prevent log or div by zero.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        # Mask for invalid values (-inf) in p or q
        mask_p_invalid = torch.isinf(p) & (p == -float('inf'))
        mask_q_invalid = torch.isinf(q) & (q == -float('inf'))

        # Apply softmax and log_softmax with numerical stability
        p_log_softmax = torch.log_softmax(p, dim=-1)
        q_softmax = torch.softmax(q, dim=-1)

        # Compute the cross-entropy loss with masking applied directly during the summation
        loss = -q_softmax * p_log_softmax

        # Mask out invalid values by setting the loss of these values to zero
        loss = torch.where(mask_p_invalid | mask_q_invalid, torch.zeros_like(loss), loss)

        # Sum over the last dimension and compute the mean
        loss = loss.sum(dim=-1)

        # Take the mean loss, ignoring invalid entries
        return loss.mean()

    def forward(self, student_output, teacher_output):
        """Average cross-entropy over all mismatched student/teacher view pairs."""
        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    continue
                consistency += self.my_cross_entropy(student_output[i], teacher_output[j])
                count += 1

        consistency /= count
        return consistency
