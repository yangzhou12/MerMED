import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# OPENEYE_MEAN = (0.31561512, 0.20948674, 0.16498742)
# OPENEYE_STD = (0.13639495, 0.08581575, 0.08441528)

OPENEYE_MEAN = (0.485, 0.456, 0.406)
OPENEYE_STD = (0.229, 0.224, 0.225)

_NORMS = {
    'cfp': ((0.33781426, 0.21333193, 0.13285545), (0.29643703, 0.1900187, 0.13929177)),
    'oct': ((0.19832779, 0.19832779, 0.19832777), (0.22167036, 0.22167036, 0.22167036)),
    'cxr': ((0.50756656, 0.50756656, 0.50756656), (0.31223216, 0.31223216, 0.31223216)),
    'pathology': ((0.72139378, 0.60500935, 0.70543986), (0.24855434, 0.27136655, 0.24570293)),
    'CT': ((0.33009237, 0.33009237, 0.33009237), (0.32522697, 0.32522697, 0.32522697)),
    'US': ((0.14694033, 0.14694033, 0.14694033), (0.1806314, 0.1806314, 0.1806314)),
    'skin': ((0.66138601, 0.51889024, 0.47144768), (0.23963617, 0.21994001, 0.22855408))
}

# Modality: cfp
# Mean: [0.33781426 0.21333193 0.13285545]
# Std: [0.29643703 0.1900187  0.13929177]

# Modality: oct
# Mean: [0.19832779 0.19832779 0.19832777]
# Std: [0.22167036 0.22167036 0.22167036]

# Modality: cxr
# Mean: [0.50756656 0.50756656 0.50756656]
# Std: [0.31223216 0.31223216 0.31223216]

# Modality: pathology
# Mean: [0.72139378 0.60500935 0.70543986]
# Std: [0.24855434 0.27136655 0.24570293]

# Modality: CT
# Mean: [0.33009237 0.33009237 0.33009237]
# Std: [0.32522697 0.32522697 0.32522697]

# Modality: US
# Mean: [0.14694033 0.14694033 0.14694033]
# Std: [0.1806314 0.1806314 0.1806314]

# Modality: skin
# Mean: [0.66138601 0.51889024 0.47144768]
# Std: [0.23963617 0.21994001 0.22855408]

class TSMDataset(Dataset):
    def __init__(self, csv_file, root_dir, split="train", train_size=1, transform=None, 
                 label_mapping=None, is_external_test=False):
        """
        Args:
            csv_file (string): Path to the CSV file with columns "image" and "label".
            root_dir (string): Directory with all the images.
            split (string): One of "train", "val", "test".
            train_size (float): Fraction of training data to use (for ablation studies).
            transform (callable, optional): Optional transform to be applied on a sample.
            label_mapping (dict, optional): Mapping from label names to indices. 
                                           Used for external test sets to align with training labels.
            is_external_test (bool): If True, treats this as an external test set (no splitting).
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_external_test = is_external_test
        
        # Determine if multilabel
        self.is_multilabel = False
        if "[" in str(self.data["label"].iloc[0]):
            self.is_multilabel = True
        
        # Handle label mapping
        if label_mapping is not None:
            # Use provided label mapping (for external test sets)
            self.label2idx = label_mapping
            # Filter out samples with labels not in the training set
            if not self.is_multilabel:
                valid_labels = set(self.label2idx.keys())
                original_len = len(self.data)
                self.data = self.data[self.data["label"].isin(valid_labels)].reset_index(drop=True)
                filtered_count = original_len - len(self.data)
                if filtered_count > 0:
                    print(f"Warning: Filtered out {filtered_count} samples with labels not in training set")
        else:
            # Create label mapping from this dataset (for training)
            if not self.is_multilabel:
                # label_list = sorted(self.data["label"].unique().tolist())
                label_list = self.data["label"].unique().tolist()
                self.label2idx = {name: i for i, name in enumerate(label_list)}
        
        # Handle data splitting
        if is_external_test:
            # External test set - use all data, no splitting
            self.data["split"] = "test"
        else:
            # Internal dataset - perform splitting if needed
            if "split" not in self.data.columns:
                self.split_data()
            self.data = self.data[self.data["split"] == split].reset_index(drop=True)
            
            # Apply train_size reduction if specified
            if train_size < 1 and split == "train":
                if self.is_multilabel:
                    self.data, _ = train_test_split(
                        self.data, train_size=train_size, random_state=42
                    )
                else:
                    self.data, _ = train_test_split(
                        self.data, train_size=train_size, random_state=42, 
                        stratify=self.data['label']
                    )
                self.data = self.data.reset_index(drop=True)
    
    def split_data(self, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
        """Split DataFrame into train, validation, and test sets"""
        if self.is_multilabel:
            # Simple random split for multilabel
            train_data, temp_test_data = train_test_split(
                self.data, train_size=train_size, random_state=seed
            )
            rel_val_size = val_size / (val_size + test_size)
            val_data, test_data = train_test_split(
                temp_test_data, train_size=rel_val_size, random_state=seed
            )
        else:
            # Stratified split for single-label
            train_data, temp_test_data = train_test_split(
                self.data, train_size=train_size, random_state=seed, 
                stratify=self.data['label']
            )
            rel_val_size = val_size / (val_size + test_size)
            val_data, test_data = train_test_split(
                temp_test_data, train_size=rel_val_size, random_state=seed, 
                stratify=temp_test_data['label']
            )
        
        # Add split column
        self.data['split'] = 'train'
        self.data.loc[val_data.index, 'split'] = 'val'
        self.data.loc[test_data.index, 'split'] = 'test'
    
    def get_label_mapping(self):
        """Return the label to index mapping"""
        return self.label2idx.copy()
    
    def get_num_classes(self):
        """Return the number of classes"""
        return len(self.label2idx) if not self.is_multilabel else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_point = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, data_point.image_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.is_multilabel:
            label = eval(data_point.label)
        else:
            label = self.label2idx[data_point.label]
        
        label = torch.tensor(label, dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Usage example for external validation
def create_external_datasets(args, transform=None):
    """
    Create training and external test datasets with aligned label mappings.
    
    Args:
        train_csv: Path to training CSV file
        test_csv: Path to external test CSV file
        train_root: Root directory for training images
        test_root: Root directory for test images
        transform: Image transforms to apply
    
    Returns:
        train_dataset, val_dataset, test_dataset, label_mapping
    """
    if transform == None:
        transform = build_transform("test", args)

    # Create training dataset
    train_dataset = TSMDataset(
        csv_file=args.train_label_path,
        root_dir=args.train_data_path,
        split="train",
        transform=transform
    )
    
    # Create validation dataset (from same CSV as training)
    val_dataset = TSMDataset(
        csv_file=args.train_label_path,
        root_dir=args.train_data_path,
        split="val",
        transform=transform
    )
    
    # Get label mapping from training set
    label_mapping = train_dataset.get_label_mapping()
    
    # Create external test dataset with the training label mapping
    test_dataset = TSMDataset(
        csv_file=args.test_label_path,
        root_dir=args.test_data_path,
        split="test",
        transform=transform,
        label_mapping=label_mapping,
        is_external_test=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"External test samples: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    print(f"Label mapping: {label_mapping}")
    
    return train_dataset, val_dataset, test_dataset, label_mapping

# class TSMDataset(Dataset):
#     def __init__(self, csv_file, root_dir, split="train", train_size=1, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the CSV file with columns "image" and "label".
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.data = pd.read_csv(csv_file)
#         self.is_multilabel = False
#         if "[" in str(self.data["label"][0]):
#             self.is_multilabel = True
#         else:
#             label_list = self.data["label"].unique().tolist()
#             self.label2idx = {name: i for i, name in enumerate(label_list)}

#         if "split" not in self.data.columns:
#             self.split_data()

#         self.data = self.data[self.data["split"] == split]
#         if train_size < 1:
#             if self.is_multilabel:
#                 self.data, _ = train_test_split(self.data, train_size=train_size, random_state=42)
#             else:
#                 self.data, _ = train_test_split(self.data, train_size=train_size, random_state=42, stratify=self.data['label'])

#         self.root_dir = root_dir
#         self.transform = transform

#     def split_data(self, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
#         # Split DataFrame into train, validation, and test sets
#         train_data, temp_test_data = train_test_split(self.data, train_size=train_size, random_state=seed, stratify=self.data['label'])
#         rel_val_size = val_size / (val_size + test_size)
#         val_data, test_data = train_test_split(temp_test_data, train_size=rel_val_size, random_state=seed, stratify=temp_test_data['label'])

#         # Add a new column to indicate the split
#         self.data['split'] = 'train'
#         self.data.loc[val_data.index, 'split'] = 'val'
#         self.data.loc[test_data.index, 'split'] = 'test'

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data_point = self.data.iloc[idx]
#         img_path = os.path.join(self.root_dir, data_point.image_path)
#         image = Image.open(img_path).convert('RGB')
        
#         if self.is_multilabel:
#             label = eval(data_point.label)
#         else:
#             label = self.label2idx[data_point.label]
#         label = torch.tensor(label, dtype=torch.int64)

#         if self.transform:
#             image = self.transform(image)

#         return image, label

class HFDataset(Dataset):
    def __init__(self, dataset_name, split="train", train_size=1, transform=None):
        """
        Args:
            dataset_name (string): Name of the Hugging Face dataset
            split (string): Split to use (train/validation/test)
            train_size (float): Percentage of training data to use
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Load the dataset
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform
        
        # Handle train_size
        if train_size < 1:
            indices = list(range(len(self.dataset)))
            train_indices, _ = train_test_split(indices, train_size=train_size, random_state=42)
            self.dataset = self.dataset.select(train_indices)
            
        # Create label mapping for multi-class classification
        if not self.is_multilabel():
            unique_labels = sorted(set(self.dataset['label']))
            self.label2idx = {label: i for i, label in enumerate(unique_labels)}
            
    def is_multilabel(self):
        """Check if the dataset is multi-label by examining the first label"""
        first_label = self.dataset[0]['label']
        return isinstance(first_label, (list, tuple)) or (isinstance(first_label, str) and first_label.startswith('['))
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert image to RGB if needed
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            # Handle numpy arrays or other formats
            image = Image.fromarray(image).convert('RGB')
            
        if self.is_multilabel():
            label = item['label']
            if isinstance(label, str):
                label = eval(label)
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = self.label2idx[item['label']]
            label = torch.tensor(label, dtype=torch.int64)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def build_dataset(data_split, args, train_size=1):
    transform = build_transform(data_split, args)
    
    # Check if label_path ends with .csv to determine if it's a local dataset
    is_local_dataset = args.label_path.lower().endswith('.csv')
    
    if not is_local_dataset:
        if data_split == "val":
            data_split = "validation"
        # Use Hugging Face dataset
        dataset = HFDataset(
            dataset_name=args.label_path,  # Use label_path as the dataset name
            split=data_split,
            train_size=train_size,
            transform=transform
        )
    else:
        # Use local dataset
        dataset = TSMDataset(
            csv_file=args.label_path,
            root_dir=args.data_path,
            split=data_split,
            train_size=train_size,
            transform=transform
        )
        
    return dataset


def build_transform(split, args):
    if args.modality and args.modality in _NORMS:
        mean, std = _NORMS[args.modality]
    else:
        mean = OPENEYE_MEAN
        std = OPENEYE_STD

    # train transform
    # if is_train=='train':
    if 'train' in split:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            # interpolation = InterpolationMode.BICUBIC,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        transform.transforms[0].interpolation = InterpolationMode.BICUBIC
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

