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

class TSMDataset(Dataset):
    def __init__(self, csv_file, root_dir, split="train", train_size=1, transform=None, 
                 label_mapping=None, is_external_test=False, sensitive_attr=None,
                 return_sensitive_attr=False):
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
            sensitive_attr (str or list): Column name(s) for sensitive attributes (e.g., 'age', 'gender', ['age', 'gender']).
            return_sensitive_attr (bool): If True, returns sensitive attributes along with image and label.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_external_test = is_external_test
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
            label = eval(data_point.label)
        else:
            label = self.label2idx[data_point.label]
        
        label = torch.tensor(label, dtype=torch.int64)
        
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

# Usage example for external validation
def create_external_datasets(args, transform=None):
    """
    Create training and external test datasets with aligned label mappings.
    
    Args:
        args: Arguments containing paths and configuration
        transform: Image transforms to apply
    
    Returns:
        train_dataset, val_dataset, test_dataset, label_mapping
    """
    if transform == None:
        transform = build_transform("test", args)

    # Determine sensitive attributes
    sensitive_attr = getattr(args, 'sensitive_attr', None)
    return_sensitive_attr = getattr(args, 'return_sensitive_attr', False)

    # Create training dataset
    train_dataset = TSMDataset(
        csv_file=args.train_label_path,
        root_dir=args.train_data_path,
        split="train",
        transform=transform,
        sensitive_attr=sensitive_attr,
        return_sensitive_attr=return_sensitive_attr
    )
    
    # Create validation dataset (from same CSV as training)
    val_dataset = TSMDataset(
        csv_file=args.train_label_path,
        root_dir=args.train_data_path,
        split="val",
        transform=transform,
        sensitive_attr=sensitive_attr,
        return_sensitive_attr=return_sensitive_attr
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
        is_external_test=True,
        sensitive_attr=sensitive_attr,
        return_sensitive_attr=return_sensitive_attr
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"External test samples: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    print(f"Label mapping: {label_mapping}")
    
    # Print sensitive attribute information
    if sensitive_attr is not None:
        sens_info = train_dataset.get_sensitive_attr_info()
        print(f"\nSensitive attributes: {sens_info['attributes']}")
        print(f"Sensitive attribute mappings: {sens_info['mappings']}")
        
        print("\nSensitive attribute distributions:")
        for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
            dist = dataset.get_sensitive_attr_distribution()
            print(f"\n{split_name} split:")
            for attr, values in dist.items():
                print(f"  {attr}: {values}")
    
    return train_dataset, val_dataset, test_dataset, label_mapping

class HFDataset(Dataset):
    def __init__(self, dataset_name, split="train", train_size=1, transform=None,
                 sensitive_attr=None, return_sensitive_attr=False):
        """
        Args:
            dataset_name (string): Name of the Hugging Face dataset
            split (string): Split to use (train/validation/test)
            train_size (float): Percentage of training data to use
            transform (callable, optional): Optional transform to be applied on a sample
            sensitive_attr (str or list): Column name(s) for sensitive attributes
            return_sensitive_attr (bool): If True, returns sensitive attributes along with image and label
        """
        # Load the dataset
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform
        self.return_sensitive_attr = return_sensitive_attr
        
        # Handle sensitive attributes
        self.sensitive_attr = sensitive_attr
        self.sensitive_attr_mappings = {}
        
        if self.sensitive_attr is not None:
            if isinstance(self.sensitive_attr, str):
                self.sensitive_attr = [self.sensitive_attr]
            
            # Create mappings for categorical sensitive attributes
            for attr in self.sensitive_attr:
                if attr not in self.dataset.column_names:
                    raise ValueError(f"Sensitive attribute '{attr}' not found in dataset columns: {self.dataset.column_names}")
                
                # Get unique values
                unique_values = set()
                for item in self.dataset:
                    if item[attr] is not None:
                        unique_values.add(item[attr])
                
                # Check if categorical
                if len(unique_values) > 0 and isinstance(list(unique_values)[0], str):
                    unique_values = sorted(list(unique_values))
                    self.sensitive_attr_mappings[attr] = {val: i for i, val in enumerate(unique_values)}
                else:
                    self.sensitive_attr_mappings[attr] = None
        
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
    
    def get_sensitive_attr_info(self):
        """Return information about sensitive attributes"""
        return {
            'attributes': self.sensitive_attr,
            'mappings': self.sensitive_attr_mappings.copy()
        }
        
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
        
        # Handle sensitive attributes
        if self.return_sensitive_attr and self.sensitive_attr is not None:
            sensitive_values = {}
            for attr in self.sensitive_attr:
                value = item.get(attr)
                
                if value is None:
                    sensitive_values[attr] = torch.tensor(-1, dtype=torch.int64)
                else:
                    if self.sensitive_attr_mappings[attr] is not None:
                        mapped_value = self.sensitive_attr_mappings[attr][value]
                        sensitive_values[attr] = torch.tensor(mapped_value, dtype=torch.int64)
                    else:
                        sensitive_values[attr] = torch.tensor(float(value), dtype=torch.float32)
            
            return image, label, sensitive_values
        else:
            return image, label

def build_dataset(data_split, args, train_size=1):
    transform = build_transform(data_split, args)
    
    # Check if label_path ends with .csv to determine if it's a local dataset
    is_local_dataset = args.label_path.lower().endswith('.csv')
    
    # Get sensitive attribute settings
    sensitive_attr = getattr(args, 'sensitive_attr', None)
    return_sensitive_attr = getattr(args, 'return_sensitive_attr', False)
    
    if not is_local_dataset:
        if data_split == "val":
            data_split = "validation"
        # Use Hugging Face dataset
        dataset = HFDataset(
            dataset_name=args.label_path,
            split=data_split,
            train_size=train_size,
            transform=transform,
            sensitive_attr=sensitive_attr,
            return_sensitive_attr=return_sensitive_attr
        )
    else:
        # Use local dataset
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
    if args.modality and args.modality in _NORMS:
        mean, std = _NORMS[args.modality]
    else:
        mean = OPENEYE_MEAN
        std = OPENEYE_STD

    # train transform
    if 'train' in split:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
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


if __name__ == "__main__":
    # Example usage with sensitive attributes
    data_path = "/path/to/data"
    label_path = "/path/to/labels.csv"
    
    # Single sensitive attribute
    dataset = TSMDataset(
        csv_file=label_path, 
        root_dir=data_path, 
        split="train", 
        transform=None,
        sensitive_attr="gender",  # or "age"
        return_sensitive_attr=True
    )
    
    # Multiple sensitive attributes
    dataset_multi = TSMDataset(
        csv_file=label_path, 
        root_dir=data_path, 
        split="train", 
        transform=None,
        sensitive_attr=["gender", "age"],
        return_sensitive_attr=True
    )
    
    # Get a sample
    if len(dataset) > 0:
        image, label, sensitive_attrs = dataset[0]
        print(f"Label: {label}")
        print(f"Sensitive attributes: {sensitive_attrs}")
    
    # Get sensitive attribute info
    sens_info = dataset.get_sensitive_attr_info()
    print(f"\nSensitive attribute info: {sens_info}")
    
    # Get distribution
    dist = dataset.get_sensitive_attr_distribution()
    print(f"\nSensitive attribute distribution: {dist}")