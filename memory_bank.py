import torch.nn as nn
import torch

class MemoryBank(nn.Module):
    def __init__(self, ncrops, K=65536, out_dim=256):
        # create the queue
        super().__init__()
        self.K = K
        self.ncrops = ncrops
        self.register_buffer("queue", torch.randn(out_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # Add label queue
        self.register_buffer("label_queue", -1 * torch.ones(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @staticmethod
    @torch.no_grad()
    def concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def update_queue(self, keys, labels):
        # gather keys and labels before updating queue
        keys = self.concat_all_gather(keys)
        labels = self.concat_all_gather(labels)
        
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f"{self.K} % {batch_size}"  # for simplicity

        # replace the keys and labels at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.label_queue[ptr:ptr + batch_size] = labels
        
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def get_features(self):
        return self.queue.clone().detach()
    
    def get_labels(self):
        return self.label_queue.clone().detach()

    def forward(self, x, y, update=False):
        # Compute logits
        logits = x @ self.queue.clone().detach()
        
        if -1 not in self.label_queue:
            # Create mask where True indicates different labels
            mask = y.repeat(x.size(0)//y.size(0)).unsqueeze(1) != self.label_queue.unsqueeze(0)  
            
            # Mask logits where labels differ (set to large negative value)
            # logits[mask] = -1e3
            logits[mask] = -float('inf')
        
        if update:
            x2 = x.detach().chunk(2)[1]
            y2 = y.detach()
            self.update_queue(x2, y2)
            
        return logits