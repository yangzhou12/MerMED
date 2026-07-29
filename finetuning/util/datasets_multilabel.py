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

# OPENEYE_MEAN = (0.31561512, 0.20948674, 0.16498742)
# OPENEYE_STD = (0.13639495, 0.08581575, 0.08441528)

OPENEYE_MEAN = (0.485, 0.456, 0.406)
OPENEYE_STD = (0.229, 0.224, 0.225)

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

# Define normalization constants for each modality
_NORMS = {
    'cfp': ((0.33781426, 0.21333193, 0.13285545), (0.29643703, 0.1900187, 0.13929177)),
    'oct': ((0.19832779, 0.19832779, 0.19832777), (0.22167036, 0.22167036, 0.22167036)),
    'cxr': ((0.50756656, 0.50756656, 0.50756656), (0.31223216, 0.31223216, 0.31223216)),
    'pathology': ((0.72139378, 0.60500935, 0.70543986), (0.24855434, 0.27136655, 0.24570293)),
    'CT': ((0.33009237, 0.33009237, 0.33009237), (0.32522697, 0.32522697, 0.32522697)),
    'US': ((0.14694033, 0.14694033, 0.14694033), (0.1806314, 0.1806314, 0.1806314)),
    'skin': ((0.66138601, 0.51889024, 0.47144768), (0.23963617, 0.21994001, 0.22855408))
}

class TSMDataset(Dataset):
    def __init__(self, csv_file, root_dir, split="train", train_size=1, transform=None,
                 sensitive_attr=None, return_sensitive_attr=False):
        """
        Args:
            csv_file (string): Path to the CSV file with columns "image_path" and either:
                - "label" for multi-class classification
                - "label_multihot" for multi-label classification
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            sensitive_attr (str or list): Column name(s) for sensitive attributes (e.g., 'age', 'gender', ['age', 'gender']).
            return_sensitive_attr (bool): If True, returns sensitive attributes along with image and label.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.return_sensitive_attr = return_sensitive_attr
        
        # Handle sensitive attributes
        self.sensitive_attr = sensitive_attr
        self.sensitive_attr_mappings = {}
        
        if self.sensitive_attr is not None:
            # Convert to list if single attribute
            if isinstance(self.sensitive_attr, str):
                self.sensitive_attr = [self.sensitive_attr]
            
            # Verify sensitive attributes exist in the CSV
            for attr in self.sensitive_attr:
                if attr not in self.data.columns:
                    raise ValueError(f"Sensitive attribute '{attr}' not found in CSV columns: {self.data.columns.tolist()}")
            
            # Create mappings for categorical sensitive attributes
            for attr in self.sensitive_attr:
                # Check if attribute is categorical (string) or numerical
                if self.data[attr].dtype == 'object' or self.data[attr].dtype.name == 'category':
                    unique_values = sorted(self.data[attr].dropna().unique().tolist())
                    self.sensitive_attr_mappings[attr] = {val: i for i, val in enumerate(unique_values)}
                else:
                    # For numerical attributes, no mapping needed
                    self.sensitive_attr_mappings[attr] = None
        
        # Determine if this is multi-label classification
        # First check for label_multihot column, then fallback to checking label format
        self.is_multilabel = 'label_multihot' in self.data.columns
        if not self.is_multilabel and 'label' in self.data.columns and '[' in str(self.data["label"].iloc[0]):
            self.is_multilabel = True
            # Rename label column to label_multihot for consistency
            self.data = self.data.rename(columns={'label': 'label_multihot'})
        
        if not self.is_multilabel:
            # For multi-class, create label mapping
            label_list = self.data["label"].unique().tolist()
            self.label2idx = {name: i for i, name in enumerate(label_list)}

        if "split" not in self.data.columns:
            self.split_data()

        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        if train_size < 1 and split == "train":
            # For multi-label, we don't use stratification since it's not appropriate
            # for multiple labels per sample
            self.data, _ = train_test_split(
                self.data, 
                train_size=train_size, 
                random_state=42,
                stratify=None if self.is_multilabel else self.data['label']
            )
            self.data = self.data.reset_index(drop=True)

    def split_data(self, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
        # Split DataFrame into train, validation, and test sets
        stratify_col = 'label' if not self.is_multilabel else None
        train_data, temp_test_data = train_test_split(
            self.data, train_size=train_size, random_state=seed, 
            stratify=self.data[stratify_col] if stratify_col else None
        )
        rel_val_size = val_size / (val_size + test_size)
        val_data, test_data = train_test_split(
            temp_test_data, train_size=rel_val_size, random_state=seed,
            stratify=temp_test_data[stratify_col] if stratify_col else None
        )

        # Add a new column to indicate the split
        self.data['split'] = 'train'
        self.data.loc[val_data.index, 'split'] = 'val'
        self.data.loc[test_data.index, 'split'] = 'test'

    def get_sensitive_attr_info(self):
        """Return information about sensitive attributes"""
        return {
            'attributes': self.sensitive_attr,
            'mappings': self.sensitive_attr_mappings.copy()
        }
    
    def get_sensitive_attr_distribution(self):
        """Get distribution of sensitive attributes in the dataset"""
        if self.sensitive_attr is None:
            return None
        
        distribution = {}
        for attr in self.sensitive_attr:
            distribution[attr] = self.data[attr].value_counts().to_dict()
        
        return distribution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, data_point.image_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.is_multilabel:
            # Handle both string list and string array representations
            label_str = data_point.label_multihot
            if isinstance(label_str, str):
                label = eval(label_str)
            else:
                label = label_str
            label = torch.tensor(label, dtype=torch.float32)  # Use float32 for multi-label
        else:
            label = self.label2idx[data_point.label]
            label = torch.tensor(label, dtype=torch.int64)  # Use int64 for multi-class

        if self.transform:
            image = self.transform(image)
        
        # Handle sensitive attributes
        if self.return_sensitive_attr and self.sensitive_attr is not None:
            sensitive_values = {}
            for attr in self.sensitive_attr:
                value = data_point[attr]
                
                # Handle missing values
                if pd.isna(value):
                    sensitive_values[attr] = torch.tensor(-1, dtype=torch.int64)  # Use -1 for missing
                else:
                    # Apply mapping if categorical, otherwise use numerical value
                    if self.sensitive_attr_mappings[attr] is not None:
                        mapped_value = self.sensitive_attr_mappings[attr][value]
                        sensitive_values[attr] = torch.tensor(mapped_value, dtype=torch.int64)
                    else:
                        # Numerical attribute
                        sensitive_values[attr] = torch.tensor(float(value), dtype=torch.float32)
            
            return image, label, sensitive_values
        else:
            return image, label

def build_dataset(data_split, args, train_size=1):
    transform = build_transform(data_split, args)
    
    # Get sensitive attribute settings
    sensitive_attr = getattr(args, 'sensitive_attr', None)
    return_sensitive_attr = getattr(args, 'return_sensitive_attr', False)

    dataset = TSMDataset(
        csv_file=args.label_path, 
        root_dir=args.data_path, 
        split=data_split, 
        train_size=train_size, 
        transform=transform,
        sensitive_attr=sensitive_attr,
        return_sensitive_attr=return_sensitive_attr
    )
    return dataset


def build_transform(split, args):
    # Use modality-specific normalization if specified, otherwise use default
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
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC)
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

