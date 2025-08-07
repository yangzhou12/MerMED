import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

class MerMEDDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with columns "image" and "label".
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.modality2idx = {label: idx for idx, label in enumerate(sorted(self.df["modality"].unique()))}
        
        # Initialize tracking variables
        self.target_indices = []
        min_class_size = float('inf')  # More explicit than None
        
        # Build indices for each class
        for class_name, class_idx in self.modality2idx.items():
            indices = self.df[self.df["modality"] == class_name].index.tolist()
            self.target_indices.append(indices)
            min_class_size = min(min_class_size, len(indices))
            
        self.min_class_size = min_class_size  # Store for potential future use
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_point = self.df.iloc[idx]
        img_path = data_point.image_path
        image = Image.open(img_path)
        modality = self.modality2idx[data_point.modality]
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image, data_point.modality)

        return image, modality

import torch
import numpy as np
from torch.utils.data import Sampler


class InfiniteClassSampler:
    """
    An infinite iterator over a fixed list of indices, with periodic reshuffling.
    Sampling 'with replacement' from a (potentially small) local subset of a class.
    """
    def __init__(self, indices, base_seed=0):
        """
        Args:
            indices (array-like): The local subset of sample indices for this class.
            base_seed (int): Base seed to ensure reproducible shuffle cycles.
        """
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.base_seed = base_seed

    def __iter__(self):
        """
        Yields an infinite stream of indices in random order. 
        Once the entire list is used, reshuffle and start again.
        """
        g = torch.Generator()
        cycle_seed = self.base_seed
        while True:
            g.manual_seed(cycle_seed)
            perm = torch.randperm(len(self.indices), generator=g)
            for idx in perm:
                yield self.indices[idx].item()
            cycle_seed += 1  # increment seed so next cycle is shuffled differently

    def __len__(self):
        # The underlying subset size (though it's used infinitely).
        return len(self.indices)


class AllClassesImbalancedSampler(Sampler):
    """
    A sampler that:
      1) Uses **all classes** in each mini-batch.
      2) Addresses class imbalance by sampling each class *with replacement*.
      3) Improves efficiency via **chunked sampling** (rather than round-robin 1-by-1).
      4) Defines 'batches_per_epoch' based on the **maximum** local data size among all classes.

    Each mini-batch:
      - We compute how many samples each class should provide (balanced approach).
      - We fetch that many samples at once from an infinite iterator for each class.
      - Combine them into a single batch and yield.

    Distributed Setting:
      - Each rank sees only a slice of each class’s data, turned into an infinite iterator.
      - No class is ever “exhausted,” no matter how small (with replacement).

    Epochs:
      - We define how many batches per epoch by finding the **max** local data size among 
        all classes, then dividing by 'batch_size'. That ensures the class with the biggest 
        local partition drives how long we sample in an epoch.

    Args:
        data_source: 
            A dataset with:
              - data_source.classes = list of class labels
              - data_source.target_indices[class_id] = indices belonging to that class
        world_size (int):
            Number of ranks in distributed training.
        rank (int):
            This rank's ID in [0..world_size-1].
        batch_size (int):
            Total samples in each mini-batch.
        epochs (int):
            Number of epochs to iterate. Each epoch yields 'batches_per_epoch' batches.
        seed (int):
            Global seed for controlling any shuffle/partition logic.
    """

    def __init__(
        self,
        data_source,
        world_size: int,
        rank: int,
        batch_size: int = 8,
        epochs: int = 1,
        seed: int = 0
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = len(data_source.modality2idx)

        self.outer_epoch = 0  # can be set externally if you want to incorporate global epoch in seed

        # Base seeds
        self.base_seed = seed
        self.seed = seed  # we can increment if we want different permutations each epoch

        # 1) Compute local_class_size for each class on this rank
        local_class_sizes = []
        for class_id in range(self.num_classes):
            total_class_samples = len(self.data_source.target_indices[class_id])
            local_class_size = total_class_samples // self.world_size
            local_class_sizes.append(local_class_size)

        self.batches_per_epoch = sum(local_class_sizes) // self.batch_size

    def set_epoch(self, epoch: int):
        """
        Set the 'outer_epoch' if you want to incorporate the global epoch into your seeding.
        """
        self.outer_epoch = epoch

    def _build_infinite_class_samplers(self, epoch: int):
        """
        For each class, create an infinite sampler over that rank's local subset of the class.
        Each is seeded by a combination of the base_seed, epoch, rank, etc.
        """
        combined_seed = (
            self.base_seed
            + epoch
            + self.epochs * self.rank
            + self.outer_epoch * self.epochs * self.world_size
        )

        samplers = []
        for class_id in range(self.num_classes):
            # Partition the class among the ranks
            all_indices = self.data_source.target_indices[class_id]
            chunk_size = len(all_indices) // self.world_size
            start_idx = self.rank * chunk_size
            end_idx = start_idx + chunk_size
            local_indices = all_indices[start_idx:end_idx]

            sampler = InfiniteClassSampler(
                indices=local_indices,
                base_seed=combined_seed + class_id
            )
            samplers.append(iter(sampler))
        return samplers

    def __iter__(self):
        """
        Yields mini-batches: for each epoch, we produce 'batches_per_epoch' batches.
        Each batch is formed by chunked sampling from *all classes*.
        """
        if self.batches_per_epoch == 0:
            return  # no valid batches

        for epoch in range(self.epochs):
            class_samplers = self._build_infinite_class_samplers(epoch)

            for _ in range(self.batches_per_epoch):
                # Balanced chunk-based approach:
                # 1) Each class contributes at least `base = batch_size // num_classes` samples.
                # 2) The leftover (batch_size % num_classes) is distributed among
                #    a random subset of classes (or first leftover classes).

                base = self.batch_size // self.num_classes
                leftover = self.batch_size % self.num_classes

                # Optionally shuffle which classes get the leftover
                g = torch.Generator()
                # incorporate epoch in the leftover seed for reproducibility
                g.manual_seed(self.seed + epoch)  
                class_order = torch.randperm(self.num_classes, generator=g).tolist()

                batch_indices = []

                for c in range(self.num_classes):
                    num_to_draw = base + (1 if c in class_order[:leftover] else 0)
                    if num_to_draw > 0:
                        chunk = [next(class_samplers[c]) for _ in range(num_to_draw)]
                        batch_indices.extend(chunk)

                yield batch_indices

    def __len__(self):
        """
        Total number of mini-batches = epochs * batches_per_epoch.
        """
        return self.epochs * self.batches_per_epoch

def make_labels_matrix(
    num_classes,
    s_batch_size,
    world_size,
    unique_classes=False,
    smoothing=0.0
):
    """
    Make one-hot labels matrix for labeled samples

    NOTE: Assumes labeled data is loaded with ClassStratifiedSampler from
          src/data_manager.py
    """

    local_images = s_batch_size*num_classes
    total_images = local_images*world_size

    off_value = smoothing/(num_classes*world_size) if unique_classes else smoothing/num_classes

    if unique_classes:
        labels = torch.zeros(total_images, num_classes*world_size).cuda() + off_value
        for r in range(world_size):
            # -- index range for rank 'r' images
            s1 = r * local_images
            e1 = s1 + local_images
            # -- index offset for rank 'r' classes
            offset = r * num_classes
            for i in range(num_classes):
                labels[s1:e1][i::num_classes][:, offset+i] = 1. - smoothing + off_value
    else:
        labels = torch.zeros(total_images, num_classes).cuda() + off_value
        for i in range(num_classes):
            labels[i::num_classes][:, i] = 1. - smoothing + off_value

    return labels

def test_dataloader_iteration(csv_path):
    """
    Test iterating through the dataset using DataLoader
    """

    # Sample transform for testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.31561512, 0.20948674, 0.16498742],
            std=[0.13639495, 0.08581575, 0.08441528]
        )
    ])

    # Initialize the dataset
    dataset = MerMEDDataset(csv_file=csv_path, transform=test_transform)
    
    supervised_sampler = AllClassesImbalancedSampler(
        data_source=dataset,
        world_size=1,
        rank=0,
        batch_size=32,
        epochs=1,
        seed=0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=supervised_sampler,
        num_workers=1)

    num_epochs = 10
    for epoch in range(num_epochs):
        supervised_sampler.set_epoch(epoch)
        # Iterate through dataloader with tqdm
        for batch_images, labels in tqdm(dataloader, 
                                desc="Processing Images", 
                                total=len(dataloader),
                                unit='batch'):
            # Process batch images if needed
            # For this example, we'll just pass
            pass


# Optional: If you want to run these tests directly
if __name__ == "__main__":
    csv_path = "/mnt/workspace/zy/MedFM/MedFM_labels.csv"
    test_dataloader_iteration(csv_path)