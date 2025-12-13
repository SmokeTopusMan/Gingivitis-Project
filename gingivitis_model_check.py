import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
def setup_dist():
    """
    Initialize DDP if launched with torchrun/SLURM; returns (is_dist, rank, local_rank, world_size, device).
    Uses env:// init so it works with torchrun or SLURM-exported MASTER_*.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1

    if not is_dist:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    # Required env when using torchrun (set automatically):
    # RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    # Initialize the default process group (NCCL for multi-GPU CUDA)
    dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}")
    return True, rank, local_rank, world_size, device

IS_DIST, RANK, LOCAL_RANK, WORLD_SIZE, DEVICE = setup_dist()
IS_MAIN = (RANK == 0)
def is_pure_black_rgb(crop_rgb, thr=3, min_fraction=0.999):
    """
    Return True if at least  of pixels have all channels <= thr.
    crop_rgb: HxWx3 uint8
    thr: intensity threshold (0..255). 3 is tolerant to tiny noise.
    """
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        return False
    # pixel is "black" if all channels <= thr
    black_pixels = np.all(crop_rgb <= thr, axis=2)
    frac_black = black_pixels.mean()
    return frac_black >= min_fraction

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list=None, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        if file_list is None:
            # validation / normal usage
            self.files = sorted(os.listdir(images_dir))
        else:
            # balanced training usage
            self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        if self.transform:
            image_np = np.array(image)
            mask_np  = (np.array(mask) > 127).astype("float32")
            data = self.transform(image=image_np, mask=mask_np)
            return data["image"], data["mask"]

        return image, mask

def create_gaussian_weight_map(crop_size, sigma=None):
    """Create a 2D Gaussian weight map that gives higher weights to center pixels"""
    if sigma is None:
        sigma = crop_size / 6  # Adjust this for more/less aggressive weighting

    # Create coordinate grids
    x = np.arange(crop_size)
    y = np.arange(crop_size)
    x, y = np.meshgrid(x, y)

    # Center coordinates
    cx, cy = crop_size // 2, crop_size // 2

    # Gaussian weight map
    weight_map = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return weight_map.astype(np.float32)

def reverse_tta_predictions(predictions, transform_name):
    """Reverse TTA transformations on predictions"""
    if 'hflip' in transform_name:
        predictions = torch.flip(predictions, dims=[3])  # Flip width
    if 'vflip' in transform_name:
        predictions = torch.flip(predictions, dims=[2])  # Flip height
    return predictions

def process_crop_batch(crops, model, device, transform, transform_name='base'):
    """Process a batch of crops through the model"""
    batch_tensors = []

    for crop in crops:
        augmented = transform(image=crop, mask=np.zeros((crop.shape[0], crop.shape[1])))
        batch_tensors.append(augmented["image"])

    # Stack into batch tensor
    batch_tensor = torch.stack(batch_tensors).to(device)

    # Forward pass
    with torch.no_grad():
        predictions = model(batch_tensor)
        predictions = torch.sigmoid(predictions)

        # Handle TTA reverse transforms
        if transform_name != 'base':
            predictions = reverse_tta_predictions(predictions, transform_name)

        # Move to CPU and convert to numpy
        predictions = predictions.squeeze(1).cpu().numpy()

    return predictions

def predict_large_image(image, model, device, crop_size=256, stride=64,
                       batch_size=4, threshold=0.5, use_tta=False,
                       use_gaussian_weights=True, compute_crop_dice=False,
                       ground_truth_mask=None,
                       black_thr=3, black_min_fraction=0.999,
                       force_black_pred=True):
    """
    Improved sliding window prediction with:
    - Batch processing for efficiency
    - Gaussian weighting for better blending
    - Test Time Augmentation (TTA) option
    - Memory-efficient processing
    - Optional crop-level Dice score computation
    - Black crop handling with configurable threshold
    """

    image_np = np.array(image)
    original_h, original_w = image_np.shape[:2]

    # Calculate padding
    pad_h = (stride - (original_h - crop_size) % stride) % stride if original_h > crop_size else max(0, crop_size - original_h)
    pad_w = (stride - (original_w - crop_size) % stride) % stride if original_w > crop_size else max(0, crop_size - original_w)

    # Pad image with reflection for more natural boundaries
    padded_img = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    padded_h, padded_w = padded_img.shape[:2]

    # Pad ground truth mask if provided for crop Dice computation
    padded_gt_mask = None
    if compute_crop_dice and ground_truth_mask is not None:
        gt_mask_np = np.array(ground_truth_mask)
        padded_gt_mask = np.pad(gt_mask_np, ((0, pad_h), (0, pad_w)), mode='reflect')

    # Initialize output arrays
    pred_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)

    # Create weight map for blending
    if use_gaussian_weights:
        weight_map = create_gaussian_weight_map(crop_size)
    else:
        weight_map = np.ones((crop_size, crop_size), dtype=np.float32)

    # Create transforms
    base_transform = A.Compose([
        A.Resize(crop_size, crop_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    # TTA transforms if enabled
    tta_transforms = {}
    if use_tta:
        tta_transforms = {
            'hflip': A.Compose([A.HorizontalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
            'vflip': A.Compose([A.VerticalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
            'hvflip': A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
        }

    # Generate crop coordinates
    crop_coords = []
    for i in range(0, padded_h - crop_size + 1, stride):
        for j in range(0, padded_w - crop_size + 1, stride):
            crop_coords.append((i, j))

    print(f"Processing {len(crop_coords)} crops in batches of {batch_size}")

    # For crop-level Dice tracking
    crop_dice_scores = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(crop_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(crop_coords))
            batch_coords = crop_coords[batch_start:batch_end]

            batch_crops = []
            batch_gt_crops = []
            batch_is_black = []

            for i, j in batch_coords:
                crop = padded_img[i:i+crop_size, j:j+crop_size, :]
                batch_crops.append(crop)
                batch_is_black.append(is_pure_black_rgb(crop, thr=black_thr, min_fraction=black_min_fraction))

                if compute_crop_dice and padded_gt_mask is not None:
                    gt_crop = padded_gt_mask[i:i+crop_size, j:j+crop_size]
                    batch_gt_crops.append(gt_crop)

            # Base predictions
            batch_predictions = process_crop_batch(batch_crops, model, device, base_transform, 'base')

            # TTA
            if use_tta:
                tta_predictions = []
                for tta_name, tta_transform in tta_transforms.items():
                    tta_pred = process_crop_batch(batch_crops, model, device, tta_transform, tta_name)
                    tta_predictions.append(tta_pred)
                all_predictions = [batch_predictions] + tta_predictions
                batch_predictions = np.mean(all_predictions, axis=0)

            # OPTIONAL: force predictions to negative on black crops
            if force_black_pred:
                for k, is_blk in enumerate(batch_is_black):
                    if is_blk:
                        batch_predictions[k].fill(0.0)

            # Crop-level Dice (override GT to negative on black crops)
            if compute_crop_dice and batch_gt_crops:
                for idx, (i, j) in enumerate(batch_coords):
                    pred_crop = batch_predictions[idx]
                    gt_crop   = batch_gt_crops[idx]

                    pred_binary = (pred_crop > threshold).astype(np.float32)

                    if batch_is_black[idx]:
                        gt_binary = np.zeros_like(pred_binary, dtype=np.float32)
                    else:
                        gt_binary = (gt_crop > 127).astype(np.float32)

                    intersection = (pred_binary * gt_binary).sum()
                    union = pred_binary.sum() + gt_binary.sum()
                    dice = 1.0 if union == 0 else (2.0 * intersection) / (union + 1e-7)

                    crop_dice_scores.append({
                        'batch': batch_start // batch_size,
                        'crop_idx': batch_start + idx,
                        'position': (i, j),
                        'dice': dice,
                        'mask_coverage': gt_binary.mean(),
                        'pred_coverage': pred_binary.mean(),
                        'is_black_image_crop': bool(batch_is_black[idx]),
                    })

            # Accumulate predictions with weighting
            for idx, (i, j) in enumerate(batch_coords):
                pred = batch_predictions[idx]
                pred_sum[i:i+crop_size, j:j+crop_size] += pred * weight_map
                weight_sum[i:i+crop_size, j:j+crop_size] += weight_map

            # Progress printing
            if (batch_start // batch_size + 1) % 10 == 0:
                progress = (batch_end / len(crop_coords)) * 100
                if compute_crop_dice and crop_dice_scores:
                    recent_dice = [c['dice'] for c in crop_dice_scores[-len(batch_coords):]]
                    avg_recent_dice = np.mean(recent_dice)
                    print(f"  Progress: {progress:.1f}%, Recent crop Dice: {avg_recent_dice:.3f}")
                else:
                    print(f"  Progress: {progress:.1f}%")

    # Crop Dice stats
    if compute_crop_dice and crop_dice_scores:
        dice_values = [c['dice'] for c in crop_dice_scores]
        print(f"\n  Crop-level statistics:")
        print(f"    Total crops: {len(crop_dice_scores)}")
        print(f"    Dice scores - Mean: {np.mean(dice_values):.3f}, Std: {np.std(dice_values):.3f}")
        print(f"    Dice scores - Min: {np.min(dice_values):.3f}, Max: {np.max(dice_values):.3f}")

        best_crop = max(crop_dice_scores, key=lambda x: x['dice'])
        worst_crop = min(crop_dice_scores, key=lambda x: x['dice'])
        print(f"    Best crop: Dice={best_crop['dice']:.3f} at position {best_crop['position']}")
        print(f"    Worst crop: Dice={worst_crop['dice']:.3f} at position {worst_crop['position']}")

        high_mask = [c for c in crop_dice_scores if c['mask_coverage'] > 0.1]
        low_mask = [c for c in crop_dice_scores if c['mask_coverage'] <= 0.01]

        if high_mask:
            high_mask_dice = np.mean([c['dice'] for c in high_mask])
            print(f"    High mask crops (>10% mask): {len(high_mask)} crops, avg Dice: {high_mask_dice:.3f}")

        if low_mask:
            low_mask_dice = np.mean([c['dice'] for c in low_mask])
            print(f"    Low mask crops (≤1% mask): {len(low_mask)} crops, avg Dice: {low_mask_dice:.3f}")

    # Normalize by weights
    final_pred = np.divide(pred_sum, weight_sum, out=np.zeros_like(pred_sum), where=weight_sum!=0)

    # Apply threshold and crop to original size
    final_mask = (final_pred > threshold).astype(np.uint8) * 255
    final_mask = final_mask[:original_h, :original_w]
    orig_np = np.array(image)  # original HxWx3
    orig_black = np.all(orig_np <= black_thr, axis=2)  # same definition as crop black check

# Set mask to 0 where original image pixels are black
    final_mask[orig_black] = 0
    if compute_crop_dice:
        return final_mask, crop_dice_scores
    else:
        return final_mask

def predict_large_image_memory_efficient(image, model, device, crop_size=256, stride=64,
                                       threshold=0.5, max_memory_gb=4,compute_crop_dice=False):
    """
    Memory-efficient version for very large images
    Processes image in chunks to avoid memory overflow
    """
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # Estimate memory usage and determine chunk size
    bytes_per_pixel = 4  # float32
    estimated_memory = (h * w * bytes_per_pixel) / (1024**3)  # GB

    print(f"Estimated memory usage: {estimated_memory:.2f} GB")

    if estimated_memory > max_memory_gb:
        print(f"Using memory-efficient processing (chunks)")
        # Process in chunks
        chunk_height = int((max_memory_gb * 1024**3) / (w * bytes_per_pixel))
        chunk_height = max(chunk_height, crop_size * 2)  # Ensure minimum chunk size

        final_mask = np.zeros((h, w), dtype=np.uint8)

        for start_h in range(0, h, chunk_height - crop_size):
            end_h = min(start_h + chunk_height, h)
            print(f"Processing chunk: {start_h}-{end_h} ({end_h-start_h}x{w})")

            chunk = image_np[start_h:end_h, :, :]

            # Convert chunk back to PIL Image
            chunk_pil = Image.fromarray(chunk)
            chunk_mask = predict_large_image(
                chunk_pil, model, device, crop_size, stride, threshold=threshold
            )

            # Handle overlapping regions
            if start_h > 0:
                # Blend overlapping region
                overlap_size = crop_size
                blend_start = max(0, crop_size // 2)

                for i in range(blend_start):
                    alpha = i / blend_start
                    final_mask[start_h + i, :] = (
                        alpha * chunk_mask[i, :] +
                        (1 - alpha) * final_mask[start_h + i, :]
                    ).astype(np.uint8)

                final_mask[start_h + blend_start:end_h, :] = chunk_mask[blend_start:, :]
            else:
                final_mask[start_h:end_h, :] = chunk_mask

        return final_mask
    else:
        print("Using standard processing")
        # Use regular method
        return predict_large_image(image, model, device, crop_size, stride, threshold=threshold)

def validate_model(model, val_loader, loss_fn, device, thr=0.5):
    """
    Validation over the full val_loader.

    Returns:
      avg_val_loss,
      avg_dice_all, avg_dice_pos,
      avg_iou_all,  avg_iou_pos,
      val_map_all,  # image-level AP over all val images (only meaningful on RANK 0)
      val_pos_count
    """
    model.eval()
    val_loss_sum = 0.0
    n_batches    = 0

    dice_all_sum = 0.0
    dice_pos_sum = 0.0
    iou_all_sum  = 0.0
    iou_pos_sum  = 0.0
    pos_img_count = 0

    all_scores = []  # for AP (only really used on RANK 0)
    all_labels = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True).unsqueeze(1).float()

            logits = model(images)
            loss   = loss_fn(logits, masks)   # DiceLoss("binary") expects logits

            val_loss_sum += float(loss.item())
            n_batches    += 1

            (dice_all_b, dice_pos_b,
             iou_all_b,  iou_pos_b,
             num_pos_b,
             scores_b, labels_b) = compute_batch_segmentation_metrics(logits, masks, thr=thr)

            dice_all_sum += float(dice_all_b.item())
            iou_all_sum  += float(iou_all_b.item())

            if num_pos_b > 0:
                dice_pos_sum += float(dice_pos_b.item())
                iou_pos_sum  += float(iou_pos_b.item())
            pos_img_count += int(num_pos_b)

            # For AP – only really need on rank 0
            if RANK == 0:
                all_scores.append(scores_b)
                all_labels.append(labels_b)

    # DDP reduce scalars
    val_loss_sum  = ddp_all_reduce_scalar(val_loss_sum, device)
    n_batches     = ddp_all_reduce_int(n_batches, device)
    dice_all_sum  = ddp_all_reduce_scalar(dice_all_sum, device)
    dice_pos_sum  = ddp_all_reduce_scalar(dice_pos_sum, device)
    iou_all_sum   = ddp_all_reduce_scalar(iou_all_sum, device)
    iou_pos_sum   = ddp_all_reduce_scalar(iou_pos_sum, device)
    pos_img_count = ddp_all_reduce_int(pos_img_count, device)

    avg_val_loss = val_loss_sum / max(1, n_batches)
    avg_dice_all = dice_all_sum / max(1, n_batches)
    avg_iou_all  = iou_all_sum  / max(1, n_batches)

    if pos_img_count > 0:
        avg_dice_pos = dice_pos_sum / pos_img_count
        avg_iou_pos  = iou_pos_sum  / pos_img_count
    else:
        avg_dice_pos = 0.0
        avg_iou_pos  = 0.0

    # AP only on main rank (others return 0.0; used only for logging/plots)
    if RANK == 0 and len(all_scores) > 0:
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        val_map_all = average_precision_from_scores(scores, labels)
    else:
        val_map_all = 0.0

    return (avg_val_loss,
            avg_dice_all, avg_dice_pos,
            avg_iou_all,  avg_iou_pos,
            val_map_all,
            pos_img_count)


def plot_metric_history(name, train_all, val_all, train_pos, val_pos, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(train_all, label=f"Train {name} (all)")
    plt.plot(val_all,   label=f"Val {name} (all)")
    if train_pos is not None and val_pos is not None:
        plt.plot(train_pos, label=f"Train {name} (pos)")
        plt.plot(val_pos,   label=f"Val {name} (pos)")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(f"{name} over epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_histories():
    # Dice
    plot_metric_history(
        "Dice",
        train_dice_all_hist, val_dice_all_hist,
        train_pos_dice_scores, val_pos_dice_scores,
        "history_dice.png"
    )
    # IoU
    plot_metric_history(
        "IoU",
        train_iou_all_hist, val_iou_all_hist,
        train_iou_pos_hist, val_iou_pos_hist,
        "history_iou.png"
    )
    # mAP (only all-images curves make sense)
    plot_metric_history(
        "mAP",
        train_map_all_hist, val_map_all_hist,
        None, None,
        "history_map.png"
    )



# Test GPU functionality
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    test_tensor = torch.randn(1000, 1000).cuda()
    result = torch.matmul(test_tensor, test_tensor)
    print("✓ GPU computation test passed")
    del test_tensor, result
else:
    print("my life is shit")
# Model setup
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # can be 'efficientnet-b0', etc.
    encoder_weights="imagenet",     # use "None" if training from scratch
    in_channels=3,
    classes=1,                      # binary segmentation
)
model = model.to(DEVICE)
if IS_DIST:
    model = DDP(model,device_ids=[LOCAL_RANK],output_device=LOCAL_RANK)

# Simple transform for training (since data is already augmented through cropping)
train_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2(),
], additional_targets={"mask": "mask"})

# Validation transform (no augmentation)
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2(),
], additional_targets={"mask": "mask"})

# Create full dataset
# --- DATASET SETUP ---
import cv2

# Full dataset
train_images_dir = "downloads/train/train_crops_images"
train_masks_dir  = "downloads/train/train_mask_crops_images"
val_images_dir   = "downloads/train/val_crops_images"
val_masks_dir    = "downloads/train/val_mask_crops_images"

# Simple sanity checks (optional – you can delete these if annoying)
for d in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
    if not os.path.isdir(d):
        print(f"⚠️  WARNING: directory not found: {d}")
image_files = sorted(os.listdir(train_images_dir))

def is_positive_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask is not None and mask.sum() > 0


image_files = sorted(os.listdir(train_images_dir))

positive = []
negative = []
import pickle

# ------ RANK 0: scan and balance ------
if RANK == 0:
    print("Rank 0: Scanning training masks to build balanced dataset...")

    for fname in image_files:
        mask_path = os.path.join(train_masks_dir, fname)

        if is_positive_mask(mask_path):
            positive.append(fname)
        else:
            negative.append(fname)

    print(f"Positives: {len(positive)}, Negatives: {len(negative)}")

    # balance logic
    n_pos = len(positive)
    negative_sampled = np.random.choice(negative, size=n_pos, replace=False).tolist()

    balanced_files = positive + negative_sampled
    np.random.shuffle(balanced_files)

    print(f"Balanced dataset size: {len(balanced_files)}")

    # serialize list to bytes
    payload = pickle.dumps(balanced_files)
    payload_tensor = torch.tensor(list(payload), dtype=torch.uint8, device=DEVICE)
    length_tensor = torch.tensor([payload_tensor.numel()], dtype=torch.int64, device=DEVICE)
else:
    payload_tensor = None
    length_tensor = torch.tensor([0], dtype=torch.int64, device=DEVICE)

# ------ broadcast length ------
dist.broadcast(length_tensor, src=0)
L = int(length_tensor.item())

# allocate recv buffer
if RANK != 0:
    payload_tensor = torch.empty((L,), dtype=torch.uint8, device=DEVICE)

# ------ broadcast payload ------
dist.broadcast(payload_tensor, src=0)

# ------ deserialize on ALL ranks ------
balanced_files = pickle.loads(bytes(payload_tensor.cpu().numpy().tolist()))

if RANK == 0:
    print(f"All ranks now have balanced list of {len(balanced_files)} items.")
# Train dataset (crops)

train_dataset = SegmentationDataset(
    train_images_dir,
    train_masks_dir,
    balanced_files,
    transform=train_transform
)

# Validation dataset (crops)
val_dataset = SegmentationDataset(
    val_images_dir,
    val_masks_dir,
    transform=val_transform
)

print(f"Val   dataset: {len(val_dataset)} crop images")

# ---- DATALOADERS ----
if IS_DIST:
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
device = DEVICE


# Move model to GPU
bce_loss = nn.BCEWithLogitsLoss()

def tversky_loss(logits, targets, alpha=0.7, beta=0.3, smooth=1e-7):
    """
    logits:  [N,1,H,W] raw outputs from model
    targets: [N,1,H,W] float {0,1}
    alpha: weight for FP  (bigger -> FP penalized more)
    beta:  weight for FN  (bigger -> FN penalized more)
    """
    probs = torch.sigmoid(logits)
    targets = targets

    # sums over H,W, keep batch dim
    dims = (2, 3)
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1 - targets)).sum(dim=dims)
    fn = ((1 - probs) * targets).sum(dim=dims)

    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky_index.mean()


def dice_loss_from_logits(logits, targets, smooth=1e-7):
    """
    Standard soft Dice loss on [N,1,H,W] logits and targets in {0,1}.
    """
    probs = torch.sigmoid(logits)         # [N,1,H,W]
    targets = targets                     # [N,1,H,W]

    # Sum over channel+spatial dimensions, keep batch dimension
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denom        = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    loss = 1.0 - dice.mean()
    return loss


def loss_fn(logits, targets):
    # Use pure Dice loss
    pos_weight = torch.tensor([5.0], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return bce(logits, targets)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Added weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Training setup
num_epochs = 75
best_val_dice = 0.0
patience = 500
patience_counter = 0
early_stop = False
# ===== Metrics helpers =====
def compute_batch_segmentation_metrics(logits: torch.Tensor,
                                       targets: torch.Tensor,
                                       thr: float = 0.5):
    """
    logits:  [N,1,H,W] raw logits
    targets: [N,1,H,W] float {0,1}
    Returns:
      dice_all, dice_pos, iou_all, iou_pos, num_pos_images,
      img_scores_cpu (1D np array), img_labels_cpu (1D np array)

    img_scores/img_labels are for image-level AP:
      - label = 1 if GT has any positive pixel
      - score = mean predicted foreground probability for that image
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    B = targets.shape[0]
    flat_preds   = preds.view(B, -1)
    flat_targets = targets.view(B, -1)
    flat_probs   = probs.view(B, -1)

    # per-image stats
    inter    = (flat_preds * flat_targets).sum(dim=1)
    pred_sum = flat_preds.sum(dim=1)
    targ_sum = flat_targets.sum(dim=1)

    # Dice per image
    denom_d = pred_sum + targ_sum
    dice_per_img = torch.where(
        denom_d == 0,
        torch.ones_like(denom_d),
        (2.0 * inter) / (denom_d + 1e-7)
    )

    # IoU per image
    union = pred_sum + targ_sum - inter
    iou_per_img = torch.where(
        union == 0,
        torch.ones_like(union),
        inter / (union + 1e-7)
    )

    # Positive-only subset (GT has any positive pixel)
    pos_mask = (targ_sum > 0)

    dice_all = dice_per_img.mean()
    iou_all  = iou_per_img.mean()

    if pos_mask.any():
        dice_pos = dice_per_img[pos_mask].mean()
        iou_pos  = iou_per_img[pos_mask].mean()
        num_pos  = int(pos_mask.sum().item())
    else:
        device   = logits.device
        dice_pos = torch.tensor(0.0, device=device)
        iou_pos  = torch.tensor(0.0, device=device)
        num_pos  = 0

    # Image-level scores & labels for AP
    # score = mean predicted foreground probability
    img_scores = flat_probs.mean(dim=1)              # [B]
    img_labels = (targ_sum > 0).float()              # [B]  (1 if any GT positive)

    return (
        dice_all, dice_pos,
        iou_all, iou_pos,
        num_pos,
        img_scores.detach().cpu().numpy(),
        img_labels.detach().cpu().numpy()
    )


def average_precision_from_scores(scores: np.ndarray,
                                  labels: np.ndarray) -> float:
    """
    Binary Average Precision (AP) from image-level scores & labels.
    scores: 1D array of predicted scores
    labels: 1D array of {0,1}
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    # No positives => AP undefined; return 0
    if labels.sum() == 0:
        return 0.0

    # Sort by decreasing score
    order = np.argsort(-scores)
    labels = labels[order]

    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)

    recall    = tp / (tp[-1])
    precision = tp / np.maximum(tp + fp, 1)

    # Standard AP as area under P-R curve (step-wise interpolation)
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r

    return float(ap)
def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    """
    logits: [N,1,H,W] raw logits or probabilities (0..1). If raw logits, pass through sigmoid before calling.
    targets: [N,1,H,W] float {0,1}
    Returns: (dice_all_batch_mean, dice_pos_batch_mean, num_pos_images_in_batch)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    # per-image dice
    intersection = (preds * targets).flatten(1).sum(dim=1)
    union = preds.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    dice_per_img = torch.where(union == 0, torch.ones_like(union), (2.0 * intersection) / (union + 1e-7))

    # positive-only mask (GT has any positives)
    pos_mask = (targets.flatten(1).sum(dim=1) > 0)
    if pos_mask.any():
        dice_pos = dice_per_img[pos_mask].mean()
        num_pos = int(pos_mask.sum().item())
    else:
        dice_pos = torch.tensor(0.0, device=logits.device)
        num_pos = 0

    dice_all = dice_per_img.mean()
    return dice_all, dice_pos, num_pos


def ddp_all_reduce_scalar(value: float, device: torch.device):
    """Sum-reduce a scalar across ranks (no-op if WORLD_SIZE==1). Returns the summed value (float)."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        t = torch.tensor([value], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return float(t.item())
    return float(value)


def ddp_all_reduce_int(value: int, device: torch.device):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        t = torch.tensor([value], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return int(t.item())
    return int(value)
# For plotting
# For plotting / history
train_losses = []
val_losses   = []

train_dice_all_hist = []
val_dice_all_hist   = []
train_pos_dice_scores = []
val_pos_dice_scores = []

train_iou_all_hist = []
val_iou_all_hist   = []
train_iou_pos_hist = []
val_iou_pos_hist   = []

train_map_all_hist = []   # AP all images (train, rank 0 subset)
val_map_all_hist   = []   # AP all images (val, full)


val_pos_counts = []
print("Starting training with validation...")
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



torch.cuda.set_device(LOCAL_RANK)


per_gpu_batch = 8
# Use a DistributedSampler for the training set:


# For validation you can keep a regular DataLoader, or use another DistributedSampler and
# compute metrics on rank 0 only. Example: only save/print on rank 0:
is_main = (RANK == 0)
if is_main:
    torch.save(model.module.state_dict(), "best_model.pth")  # note model.module under DDP
# --- inside epoch loop before iterating train_loader ---
train_loss_sum = 0.0
train_batches = 0
train_dice_all_sum = 0.0
train_dice_pos_sum = 0.0
train_pos_img_count = 0

def get_state_dict(m):
    return m.module.state_dict() if isinstance(m, DDP) else m.state_dict()

# remember each epoch:
best_val_metric = 0.0   # we'll use Val IoU(all) as key metric

for epoch in range(num_epochs):
    if IS_DIST:
        train_sampler.set_epoch(epoch)
    if early_stop:
        print("Early stopping triggered!")
        break

    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # -------- TRAINING PHASE --------
    model.train()
    train_loss_sum = 0.0
    train_batches  = 0

    dice_all_sum = 0.0
    dice_pos_sum = 0.0
    iou_all_sum  = 0.0
    iou_pos_sum  = 0.0
    pos_img_count = 0

    # For AP (image-level) during training – only on rank 0 (approximate)
    train_scores_epoch = []
    train_labels_epoch = []

    for b_idx, (images, masks) in enumerate(train_loader, start=1):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).unsqueeze(1).float()

        logits = model(images)
        loss   = loss_fn(logits, masks)   # Dice loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += float(loss.item())
        train_batches  += 1

        with torch.no_grad():
            (dice_all_b, dice_pos_b,
             iou_all_b,  iou_pos_b,
             num_pos_b,
             scores_b, labels_b) = compute_batch_segmentation_metrics(logits, masks, thr=0.5)

            dice_all_sum += float(dice_all_b.item())
            iou_all_sum  += float(iou_all_b.item())
            if num_pos_b > 0:
                dice_pos_sum += float(dice_pos_b.item())
                iou_pos_sum  += float(iou_pos_b.item())
            pos_img_count += int(num_pos_b)

            if RANK == 0:
                train_scores_epoch.append(scores_b)
                train_labels_epoch.append(labels_b)

        if b_idx % 10 == 0 and RANK == 0:
            print(f"  Batch {b_idx}, Loss: {loss.item():.4f}, "
                  f"Dice(all): {dice_all_b.item():.4f}, IoU(all): {iou_all_b.item():.4f}")

    # DDP reduce training scalars
    train_loss_sum  = ddp_all_reduce_scalar(train_loss_sum, device)
    train_batches   = ddp_all_reduce_int(train_batches, device)
    dice_all_sum    = ddp_all_reduce_scalar(dice_all_sum, device)
    dice_pos_sum    = ddp_all_reduce_scalar(dice_pos_sum, device)
    iou_all_sum     = ddp_all_reduce_scalar(iou_all_sum, device)
    iou_pos_sum     = ddp_all_reduce_scalar(iou_pos_sum, device)
    pos_img_count   = ddp_all_reduce_int(pos_img_count, device)

    avg_train_loss = train_loss_sum / max(1, train_batches)
    avg_dice_all   = dice_all_sum / max(1, train_batches)
    avg_iou_all    = iou_all_sum  / max(1, train_batches)

    if pos_img_count > 0:
        avg_dice_pos = dice_pos_sum / pos_img_count
        avg_iou_pos  = iou_pos_sum  / pos_img_count
    else:
        avg_dice_pos = 0.0
        avg_iou_pos  = 0.0

    # Training mAP (image-level) on rank 0 only (subset of train set due to DDP sampling)
    if RANK == 0 and len(train_scores_epoch) > 0:
        tr_scores = np.concatenate(train_scores_epoch, axis=0)
        tr_labels = np.concatenate(train_labels_epoch, axis=0)
        train_map_all = average_precision_from_scores(tr_scores, tr_labels)
    else:
        train_map_all = 0.0

    # -------- VALIDATION PHASE --------
    (avg_val_loss,
     avg_val_dice_all, avg_val_dice_pos,
     avg_val_iou_all,  avg_val_iou_pos,
     val_map_all,
     val_pos_count) = validate_model(model, val_loader, loss_fn, device, thr=0.5)

    # LR scheduler on val loss
    scheduler.step(avg_val_loss)

    # -------- LOGGING & HISTORY --------
    if RANK == 0:
        print(f"  Train Loss: {avg_train_loss:.4f}, "
              f"Train Dice(all): {avg_dice_all:.4f}, Train Dice(pos): {avg_dice_pos:.4f}, "
              f"Train IoU(all): {avg_iou_all:.4f},  Train IoU(pos): {avg_iou_pos:.4f}, "
              f"Train mAP(all): {train_map_all:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f}, "
              f"Val   Dice(all): {avg_val_dice_all:.4f}, Val   Dice(pos): {avg_val_dice_pos:.4f} ({val_pos_count} pos imgs), "
              f"Val   IoU(all): {avg_val_iou_all:.4f},  Val   IoU(pos): {avg_val_iou_pos:.4f}, "
              f"Val   mAP(all): {val_map_all:.4f}")

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    train_dice_all_hist.append(avg_dice_all)
    val_dice_all_hist.append(avg_val_dice_all)
    train_pos_dice_scores.append(avg_dice_pos)
    val_pos_dice_scores.append(avg_val_dice_pos)

    train_iou_all_hist.append(avg_iou_all)
    val_iou_all_hist.append(avg_val_iou_all)
    train_iou_pos_hist.append(avg_iou_pos)
    val_iou_pos_hist.append(avg_val_iou_pos)

    train_map_all_hist.append(train_map_all)
    val_map_all_hist.append(val_map_all)

    # Overfitting indicator
    loss_gap = avg_val_loss - avg_train_loss
    if RANK == 0:
        print(f"  Loss Gap (Val - Train): {loss_gap:.4f}")
        if loss_gap > 0.3:
            print("  ⚠️  OVERFITTING DETECTED! (Loss gap > 0.3)")
        elif loss_gap > 0.15:
            print("  ⚠️  Possible overfitting (Loss gap > 0.15)")

    # -------- CHECKPOINTING: use Val IoU(all) as key metric --------
    key_metric = avg_val_iou_all   # THIS is now your model-selection criterion

    if RANK == 0 and key_metric > best_val_metric:
        best_val_metric = key_metric
        patience_counter = 0
        torch.save(model.state_dict(), "best_model_ging.pth")  # DDP state_dict OK
        print(f"  ✓ New best Val IoU(all): {best_val_metric:.4f} - Model saved!")
    else:
        patience_counter += 1
        if RANK == 0:
            print(f"  No improvement for {patience_counter} epochs")
        if patience_counter >= patience:
            early_stop = True

    # ... run your test loop here on rank 0 only ...

if IS_DIST:
    dist.barrier()
    dist.destroy_process_group()


if RANK == 0:
    print("\nLoading best model for single-GPU testing...")

    # Use a clean device for testing (no DDP)
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fresh Unet++ model (same architecture as training)
    test_model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # weights will be loaded from checkpoint
        in_channels=3,
        classes=1,
    ).to(test_device)

    # Load checkpoint and strip "module." if present
    state = torch.load("best_model_ging.pth", map_location="cpu")

    # In case you ever save {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        cleaned_state[new_k] = v

    print("Sample keys from checkpoint:", list(state.keys())[:5])
    print("Sample keys after cleaning:", list(cleaned_state.keys())[:5])

    test_model.load_state_dict(cleaned_state, strict=True)
    test_model.eval()

    # === TESTING/EVALUATION - LOAD TEST IMAGES AT ORIGINAL SIZE ===
    print("Starting evaluation on test set...")

    class TestDatasetOriginalSize(Dataset):
        def __init__(self, images_dir, masks_dir):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.images = sorted(os.listdir(images_dir))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = self.images[idx]
            image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
            mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")
            return image, mask, img_name

    test_dataset = TestDatasetOriginalSize("downloads/test/relevant_images", "downloads/test/masks")
    print(f"Test dataset created with {len(test_dataset)} images (original sizes preserved)")

    os.makedirs("predictions", exist_ok=True)

    dice_scores_all = []
    dice_scores_pos = []
    iou_scores_all  = []
    iou_scores_pos  = []
    test_scores_for_ap  = []
    test_labels_for_ap  = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            pil_image, pil_mask, img_name = test_dataset[i]

            print(f"\nProcessing test image {i+1}/{len(test_dataset)}: {img_name}")
            print(f"ORIGINAL image size: {pil_image.size}")

            result = predict_large_image(
                pil_image, test_model, test_device,
                crop_size=256, stride=64, batch_size=4, threshold=0.5,
                use_tta=True, use_gaussian_weights=True,
                compute_crop_dice=True, ground_truth_mask=pil_mask
            )

            if isinstance(result, tuple):
                pred, crop_dice_scores = result
            else:
                pred = result
                crop_dice_scores = None

            print(f"Final prediction size: {pred.shape[1]}×{pred.shape[0]}")

            pred_tensor = torch.from_numpy(pred / 255.0).unsqueeze(0).float().to(test_device)
            save_image(pred_tensor, f"predictions/pred_{i}_{img_name}")

            mask_np = np.array(pil_mask)
            mask_tensor = torch.from_numpy((mask_np > 127).astype('float32')).unsqueeze(0).float().to(test_device)

            if mask_tensor.shape[-2:] != pred_tensor.shape[-2:]:
                print(f"⚠️  WARNING: Size mismatch! Mask: {mask_tensor.shape[-2:]}, Pred: {pred_tensor.shape[-2:]}")
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0), size=pred_tensor.shape[-2:], mode='nearest'
                ).squeeze(0)

            intersection = (pred_tensor * mask_tensor).sum()
            union_d = pred_tensor.sum() + mask_tensor.sum()
            dice_img = (2. * intersection) / (union_d + 1e-7)

            union_i = pred_tensor.sum() + mask_tensor.sum() - intersection
            if union_i > 0:
                iou_img = intersection / (union_i + 1e-7)
            else:
                iou_img = torch.tensor(1.0, device=test_device)

            dice_scores_all.append(float(dice_img.item()))
            iou_scores_all.append(float(iou_img.item()))

            gt_pos = (mask_tensor.sum() > 0)
            if gt_pos:
                dice_scores_pos.append(float(dice_img.item()))
                iou_scores_pos.append(float(iou_img.item()))

            score_img = float(pred_tensor.mean().item())
            label_img = 1.0 if gt_pos else 0.0
            test_scores_for_ap.append(score_img)
            test_labels_for_ap.append(label_img)

            print(f"Test image {i}: Dice = {dice_img.item():.4f}, IoU = {iou_img.item():.4f}")

    mean_dice_all = float(np.mean(dice_scores_all)) if dice_scores_all else 0.0
    mean_dice_pos = float(np.mean(dice_scores_pos)) if dice_scores_pos else 0.0
    mean_iou_all  = float(np.mean(iou_scores_all))  if iou_scores_all  else 0.0
    mean_iou_pos  = float(np.mean(iou_scores_pos))  if iou_scores_pos  else 0.0
    test_map_all  = average_precision_from_scores(
        np.array(test_scores_for_ap),
        np.array(test_labels_for_ap)
    ) if len(test_scores_for_ap) > 0 else 0.0

    print(f"\nFinal Results:")
    print(f"Best validation key metric (Val IoU all): {best_val_metric:.4f}")
    print(f"Average test Dice (all images):       {mean_dice_all:.4f}")
    print(f"Average test Dice (positive-only):    {mean_dice_pos:.4f}")
    print(f"Average test IoU  (all images):       {mean_iou_all:.4f}")
    print(f"Average test IoU  (positive-only):    {mean_iou_pos:.4f}")
    print(f"Test set image-level mAP (all imgs):  {test_map_all:.4f}")

    if len(val_losses) > 0:
        final_loss_gap = val_losses[-1] - train_losses[-1]
        print(f"Final training loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        print(f"Final loss gap: {final_loss_gap:.4f}")

        if final_loss_gap > 0.3:
            print("🔴 Model is overfitting significantly!")
            print("Recommendations:")
            print("- Increase data augmentation")
            print("- Add more regularization")
            print("- Reduce model complexity")
            print("- Get more training data")
        elif final_loss_gap > 0.15:
            print("🟡 Model shows some overfitting")
            print("Consider adding more regularization or data augmentation")
        else:
            print("🟢 Model generalization looks good!")


# Clear GPU cache


print("\n=== PREDICTION USAGE EXAMPLES ===")
print("For production (fast, good quality):")
print("pred = predict_large_image(image, model, device, use_tta=False, use_gaussian_weights=True)")
print("\nFor best quality (slower):")
print("pred = predict_large_image(image, model, device, use_tta=True, stride=96)")
print("\nFor very large images:")
print("pred = predict_large_image_memory_efficient(image, model, device, max_memory_gb=4)")
if IS_DIST:
    dist.destroy_process_group()
