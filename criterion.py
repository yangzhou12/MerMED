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
