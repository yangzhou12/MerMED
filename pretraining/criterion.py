"""Cross-entropy criterion between student and teacher logits for MerMED.

Supports the memory-bank-masked variant used during MerMED pretraining as well as
the plain DINO-style soft cross-entropy.

Adapted from MaSSL (https://github.com/sthalles/MaSSL).
See THIRD_PARTY_NOTICES.md for attribution and licensing status.
"""
import torch.nn as nn
import torch


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def sharpen(self, p, T):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=-1, keepdim=True)
        return sharp_p

    def support_cross_entropy(self, p, q, support_labels):
        probs = p.softmax(dim=-1) @ support_labels
        targets = q.softmax(dim=-1) @ support_labels

        probs = self.sharpen(probs, 1)
        targets = self.sharpen(targets, 1)
        # targets = self.sharpen(targets, 0.25)
        # if multicrop > 0:
        #     mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
        #     targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)

        targets[targets < 1e-4] *= 0  # numerical stability
        loss = torch.sum(torch.log(probs**(-targets)), dim=1)

        # # Step 4: compute me-max regularizer
        # rloss = 0.
        # if me_max:
        #     avg_probs = AllReduce.apply(torch.mean(sharpen(probs), dim=0))
        #     rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))

        return loss.mean()

    def cross_entropy(self, p, q):
        # assert inputs.shape == targets.shape
        # assert inputs.requires_grad == True
        # assert targets.requires_grad == False

        p = torch.log_softmax(p, dim=-1)
        q = torch.softmax(q, dim=-1)

        loss = torch.sum(-q * p, dim=-1).mean()
        return loss

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

        # # Prevent NaN or invalid values in q_softmax (re-normalize if needed)
        # q_softmax = q_softmax / (q_softmax.sum(dim=-1, keepdim=True) + epsilon)

        # Compute the cross-entropy loss with masking applied directly during the summation
        loss = -q_softmax * p_log_softmax
        
        # Mask out invalid values by setting the loss of these values to zero
        loss = torch.where(mask_p_invalid | mask_q_invalid, torch.zeros_like(loss), loss)

        # Sum over the last dimension and compute the mean
        loss = loss.sum(dim=-1)

        # Take the mean loss, ignoring invalid entries
        return loss.mean()

    def forward(self, student_output, teacher_output, support_labels=None):
        # EPS = torch.finfo(student_output[0].dtype).eps
        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    continue
                # consistency += self.cross_entropy(student_output[i], teacher_output[j])
                if support_labels is None:
                    consistency += self.my_cross_entropy(student_output[i], teacher_output[j])
                else:
                    consistency += self.support_cross_entropy(student_output[i], teacher_output[j], support_labels)
                count += 1

        consistency /= count
        return consistency

    def dino_loss(self, student_output, teacher_output):
        """
        Standard DINO loss computation.
        
        DINO loss strategy:
        - Student sees all crops: 2 global crops + local crops
        - Teacher sees only 2 global crops
        - For the 2 global student crops: match with the OTHER global teacher crop
          (student global 0 <-> teacher global 1, student global 1 <-> teacher global 0)
        - For local student crops: match with both teacher global crops
        
        Args:
            student_output: List of tensors, each of shape [batch_size, out_dim]
                - student_output[0]: first global crop
                - student_output[1]: second global crop
                - student_output[2:]: local crops
            teacher_output: List of 2 tensors, each of shape [batch_size, out_dim]
                - teacher_output[0]: first global crop
                - teacher_output[1]: second global crop
        
        Returns:
            loss: Scalar tensor with the DINO loss
        """
        total_loss = 0
        n_loss_terms = 0
        
        # Teacher has exactly 2 global crops
        assert len(teacher_output) == 2, "Teacher should have exactly 2 global crops"
        n_student_crops = len(student_output)
        
        # For the 2 global student crops: match with the OTHER teacher global crop
        # Student global 0 <-> Teacher global 1
        if n_student_crops >= 1:
            student_log_probs_0 = torch.log_softmax(student_output[0], dim=-1)
            teacher_probs_1 = torch.softmax(teacher_output[1].detach(), dim=-1)
            loss_0 = torch.sum(-teacher_probs_1 * student_log_probs_0, dim=-1).mean()
            total_loss += loss_0
            n_loss_terms += 1
        
        # Student global 1 <-> Teacher global 0
        if n_student_crops >= 2:
            student_log_probs_1 = torch.log_softmax(student_output[1], dim=-1)
            teacher_probs_0 = torch.softmax(teacher_output[0].detach(), dim=-1)
            loss_1 = torch.sum(-teacher_probs_0 * student_log_probs_1, dim=-1).mean()
            total_loss += loss_1
            n_loss_terms += 1
        
        # For local student crops: match with both teacher global crops
        for s_idx in range(2, n_student_crops):
            s_out = student_output[s_idx]
            student_log_probs = torch.log_softmax(s_out, dim=-1)
            
            # Match with teacher global 0
            teacher_probs_0 = torch.softmax(teacher_output[0].detach(), dim=-1)
            loss_local_0 = torch.sum(-teacher_probs_0 * student_log_probs, dim=-1).mean()
            total_loss += loss_local_0
            n_loss_terms += 1
            
            # Match with teacher global 1
            teacher_probs_1 = torch.softmax(teacher_output[1].detach(), dim=-1)
            loss_local_1 = torch.sum(-teacher_probs_1 * student_log_probs, dim=-1).mean()
            total_loss += loss_local_1
            n_loss_terms += 1
        
        return total_loss / n_loss_terms if n_loss_terms > 0 else torch.tensor(0.0, device=student_output[0].device)
