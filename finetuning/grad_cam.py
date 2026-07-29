"""Grad-CAM visualization for a MerMED-finetuned ViT classifier.

Loads a finetuned checkpoint, picks a few test images per class from the dataset's
label CSV, and saves per-class / overall Grad-CAM heatmaps. All paths are supplied on
the command line so the script carries no machine-specific configuration.
"""
import argparse
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
from functools import partial
import pandas as pd
import os
from pathlib import Path

OPENEYE_MEAN = (0.485, 0.456, 0.406)
OPENEYE_STD = (0.229, 0.224, 0.225)

# Determine device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Grad-CAM visualizations for a MerMED-finetuned ViT.")
parser.add_argument("--test-dataset", default="dataset", help="Dataset name, used only for output filenames.")
parser.add_argument("--model-path", required=True, help="Path to the finetuned checkpoint (checkpoint-best.pth).")
parser.add_argument("--num-classes", type=int, required=True, help="Number of classes the head was trained with.")
parser.add_argument("--label-path", required=True, help="Path to the label CSV (columns: image_path, label, split).")
parser.add_argument("--image-root", required=True, help="Root directory image_path values are resolved against.")
parser.add_argument("--top-k", type=int, default=1, help="How many classes (largest logits) to visualize.")
parser.add_argument("--samples-per-class", type=int, default=10, help="How many test images to pick per class.")
parser.add_argument("--output-dir", type=str, default="./heatmaps", help="Directory to store output images.")
args = parser.parse_args()

test_dataset = args.test_dataset
TOP_K = max(1, args.top_k)
SAMPLES_PER_CLASS = max(1, args.samples_per_class)
OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = {
    "model_path": args.model_path,
    "num_classes": args.num_classes,
    "label_path": args.label_path,
    "image_root": args.image_root,
}

model_path = cfg["model_path"]
model = VisionTransformer(
    img_size=224, num_classes=cfg["num_classes"],
    drop_path_rate=0.2,
    global_pool='avg',
    dynamic_img_size=True,
    patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    pre_norm=False, norm_layer=partial(nn.LayerNorm, eps=1e-6))

state_dict = model.state_dict()
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
checkpoint = checkpoint['model']

# load pre-trained model
msg = model.load_state_dict(checkpoint, strict=False)
print(msg)

model = model.to(device)
model.eval()

# For ViT models, we need a reshape-transform function
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# For ViT models, we need to use the blocks at the end of the transformer
# The last attention block is typically used for GradCAM with ViT
target_layers = [model.blocks[-1].norm1]  # Use the last transformer block
def select_test_images(label_df: pd.DataFrame, num_classes: int, image_root: str, samples_per_class: int = 1):
    """Return up to K test images per class based on the label file."""
    test_df = label_df[label_df["split"].str.lower() == "test"].copy()
    unique_labels = label_df["label"].unique().tolist()
    label_map = {class_name: class_idx for class_idx, class_name in enumerate(unique_labels)}
    
    selections = []
    for class_name in unique_labels:
        class_rows = test_df[test_df["label"] == class_name]
        if class_rows.empty:
            print(f"No test samples found for class {class_name}")
            continue
        for _, sample in class_rows.head(samples_per_class).iterrows():
            rel_image_path = sample["image_path"]
            full_image_path = os.path.join(image_root, rel_image_path)
            if not os.path.exists(full_image_path):
                print(f"Image not found for class {class_name}: {full_image_path}")
                continue
            selections.append(
                {
                    "class_idx": int(label_map[class_name]),
                    "class_name": class_name,
                    "image_id": str(sample.get("image_id", Path(rel_image_path).stem)),
                    "image_path": full_image_path,
                }
            )
    return selections


def run_gradcam_for_image(image_info, label_map, cam, transform, num_classes, output_dir: Path, dataset_name: str):
    """Generate Grad-CAM visualizations for a single image."""
    image_path = image_info["image_path"]
    base_name = f"{dataset_name}_{Path(image_path).stem}_class{image_info['class_name']}"
    original_img = Image.open(image_path).convert("RGB")
    rgb_img = np.array(original_img) / 255.0
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor).squeeze(0)

    probabilities = torch.softmax(logits, dim=-1)
    effective_top_k = min(TOP_K, logits.shape[0])
    top_logit_values, top_indices = torch.topk(logits, k=effective_top_k)
    top_prob_values = probabilities[top_indices]

    top_classes = [label_map.get(int(idx), f"Class {int(idx)}") for idx in top_indices.cpu().tolist()]
    top_logit_values = top_logit_values.cpu().numpy()
    top_prob_values = top_prob_values.cpu().numpy()

    all_grayscale_cams = []
    class_cams = {}
    for class_index in range(num_classes):
        targets = [ClassifierOutputTarget(class_index)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        all_grayscale_cams.append(grayscale_cam[0, :])

    overall_cam = np.mean(all_grayscale_cams, axis=0)

    for class_index in top_indices.cpu().tolist():
        targets = [ClassifierOutputTarget(int(class_index))]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        class_name = label_map.get(int(class_index), f"Class {int(class_index)}")
        class_cams[class_name] = grayscale_cam[0, :]

    if overall_cam.shape != (rgb_img.shape[0], rgb_img.shape[1]):
        from skimage.transform import resize
        overall_cam = resize(overall_cam, (rgb_img.shape[0], rgb_img.shape[1]), preserve_range=True)

    visualization = show_cam_on_image(rgb_img, overall_cam, use_rgb=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(visualization)
    plt.title(f"Overall Grad-CAM ({dataset_name})")
    plt.axis('off')
    overall_path = output_dir / f"{base_name}_overall.jpg"
    plt.savefig(str(overall_path), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {overall_path}")

    n_total_items = 2 + effective_top_k
    n_cols = 2
    n_rows = math.ceil(n_total_items / n_cols)
    fig = plt.figure(figsize=(15, n_rows * 5))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(rgb_img)
    ax_orig.set_title(f"Original Image ({dataset_name})")
    ax_orig.axis('off')

    ax_overall = fig.add_subplot(gs[0, 1])
    ax_overall.imshow(visualization)
    ax_overall.set_title(f"Overall Grad-CAM ({dataset_name})")
    ax_overall.axis('off')

    for i, (class_name, logit_value, prob) in enumerate(zip(top_classes, top_logit_values, top_prob_values)):
        pos = i + 2
        row = pos // n_cols
        col = pos % n_cols

        from skimage.transform import resize
        class_cam = class_cams[class_name]
        if class_cam.shape != (rgb_img.shape[0], rgb_img.shape[1]):
            class_cam = resize(class_cam, (rgb_img.shape[0], rgb_img.shape[1]), preserve_range=True)

        class_vis = show_cam_on_image(rgb_img, class_cam, use_rgb=True)

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(class_vis)
        ax.set_title(f"{class_name} • {dataset_name} (Logit: {logit_value:.4f}, Prob: {prob:.4f})")
        ax.axis('off')

    plt.tight_layout()
    per_class_path = output_dir / f"{base_name}_per_class.jpg"
    plt.savefig(str(per_class_path), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {per_class_path}")

    n_cols_combined = 6
    n_rows_combined = math.ceil(num_classes / n_cols_combined)
    fig_all = plt.figure(figsize=(20, 3 * n_rows_combined))
    gs_all = gridspec.GridSpec(n_rows_combined, n_cols_combined, figure=fig_all)

    from skimage.transform import resize
    for class_index in range(num_classes):
        class_name = label_map.get(class_index, f"Class {class_index}")
        row = class_index // n_cols_combined
        col = class_index % n_cols_combined
        class_cam = all_grayscale_cams[class_index]
        if class_cam.shape != (rgb_img.shape[0], rgb_img.shape[1]):
            class_cam = resize(class_cam, (rgb_img.shape[0], rgb_img.shape[1]), preserve_range=True)

        class_vis = show_cam_on_image(rgb_img, class_cam, use_rgb=True)
        ax = fig_all.add_subplot(gs_all[row, col])
        ax.imshow(class_vis)
        if class_name in top_classes:
            title = f"*{class_name}* ({dataset_name})"
        else:
            title = f"{class_name} ({dataset_name})"
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    all_class_path = output_dir / f"{base_name}_all_classes.jpg"
    plt.savefig(str(all_class_path), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {all_class_path}")


# Map class indices to readable labels
labels = pd.read_csv(cfg["label_path"])
unique_labels = labels["label"].unique().tolist()
label_map = {class_idx: class_name for class_idx, class_name in enumerate(unique_labels)}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=OPENEYE_MEAN, std=OPENEYE_STD)
])

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

selected_images = select_test_images(
    labels,
    cfg["num_classes"],
    cfg["image_root"],
    samples_per_class=SAMPLES_PER_CLASS,
)
if not selected_images:
    raise RuntimeError("No valid test images were found to process.")

for image_info in selected_images:
    print(f"Processing class {image_info['class_idx']} from {image_info['image_path']}")
    run_gradcam_for_image(
        image_info,
        label_map,
        cam,
        transform,
        cfg["num_classes"],
        OUTPUT_DIR,
        test_dataset,
    )

print("All visualizations have been saved!")