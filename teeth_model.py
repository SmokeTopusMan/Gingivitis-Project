import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")

        if self.transform:
            image_np = np.array(image)
            mask_np = (np.array(mask) > 127).astype('float32')
            augmented = self.transform(image=image_np, mask=mask_np)
            image = augmented['image']
            mask = augmented['mask']

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

def predict_large_image(image, model, device, crop_size=512, stride=128,
                       batch_size=4, threshold=0.5, use_tta=False,
                       use_gaussian_weights=True, compute_crop_dice=False,
                       ground_truth_mask=None):
    """
    Improved sliding window prediction with:
    - Batch processing for efficiency
    - Gaussian weighting for better blending
    - Test Time Augmentation (TTA) option
    - Memory-efficient processing
    - Optional crop-level Dice score computation
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
        # Process crops in batches
        for batch_start in range(0, len(crop_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(crop_coords))
            batch_coords = crop_coords[batch_start:batch_end]

            # Prepare batch
            batch_crops = []
            batch_gt_crops = []
            for i, j in batch_coords:
                crop = padded_img[i:i+crop_size, j:j+crop_size, :]
                batch_crops.append(crop)

                # Extract corresponding ground truth crop if available
                if compute_crop_dice and padded_gt_mask is not None:
                    gt_crop = padded_gt_mask[i:i+crop_size, j:j+crop_size]
                    batch_gt_crops.append(gt_crop)

            # Process base predictions
            batch_predictions = process_crop_batch(batch_crops, model, device, base_transform, 'base')

            # Process TTA if enabled
            if use_tta:
                tta_predictions = []
                for tta_name, tta_transform in tta_transforms.items():
                    tta_pred = process_crop_batch(batch_crops, model, device, tta_transform, tta_name)
                    tta_predictions.append(tta_pred)

                # Average TTA predictions
                all_predictions = [batch_predictions] + tta_predictions
                batch_predictions = np.mean(all_predictions, axis=0)

            # Compute crop-level Dice scores if requested
            if compute_crop_dice and batch_gt_crops:
                for idx, (i, j) in enumerate(batch_coords):
                    pred_crop = batch_predictions[idx]
                    gt_crop = batch_gt_crops[idx]

                    # Threshold prediction
                    pred_binary = (pred_crop > threshold).astype(np.float32)
                    gt_binary = (gt_crop > 127).astype(np.float32)

                    # Compute Dice for this crop
                    intersection = (pred_binary * gt_binary).sum()
                    union = pred_binary.sum() + gt_binary.sum()
                    dice = (2. * intersection) / (union + 1e-7)

                    crop_dice_scores.append({
                        'batch': batch_start // batch_size,
                        'crop_idx': batch_start + idx,
                        'position': (i, j),
                        'dice': dice,
                        'mask_coverage': gt_binary.mean(),  # Fraction of crop that is mask
                        'pred_coverage': pred_binary.mean()  # Fraction of crop predicted as mask
                    })

            # Accumulate predictions with weighting
            for idx, (i, j) in enumerate(batch_coords):
                pred = batch_predictions[idx]

                pred_sum[i:i+crop_size, j:j+crop_size] += pred * weight_map
                weight_sum[i:i+crop_size, j:j+crop_size] += weight_map

            # Print progress with crop Dice info
            if (batch_start // batch_size + 1) % 10 == 0:
                progress = (batch_end / len(crop_coords)) * 100
                if compute_crop_dice and crop_dice_scores:
                    recent_dice = [c['dice'] for c in crop_dice_scores[-len(batch_coords):]]
                    avg_recent_dice = np.mean(recent_dice)
                    print(f"  Progress: {progress:.1f}%, Recent crop Dice: {avg_recent_dice:.3f}")
                else:
                    print(f"  Progress: {progress:.1f}%")

    # Print crop Dice statistics if computed
    if compute_crop_dice and crop_dice_scores:
        dice_values = [c['dice'] for c in crop_dice_scores]
        print(f"\n  Crop-level statistics:")
        print(f"    Total crops: {len(crop_dice_scores)}")
        print(f"    Dice scores - Mean: {np.mean(dice_values):.3f}, Std: {np.std(dice_values):.3f}")
        print(f"    Dice scores - Min: {np.min(dice_values):.3f}, Max: {np.max(dice_values):.3f}")

        # Find best and worst performing crops
        best_crop = max(crop_dice_scores, key=lambda x: x['dice'])
        worst_crop = min(crop_dice_scores, key=lambda x: x['dice'])
        print(f"    Best crop: Dice={best_crop['dice']:.3f} at position {best_crop['position']}")
        print(f"    Worst crop: Dice={worst_crop['dice']:.3f} at position {worst_crop['position']}")

        # Analyze by mask coverage
        high_mask = [c for c in crop_dice_scores if c['mask_coverage'] > 0.1]  # >10% mask
        low_mask = [c for c in crop_dice_scores if c['mask_coverage'] <= 0.01]  # ≤1% mask

        if high_mask:
            high_mask_dice = np.mean([c['dice'] for c in high_mask])
            print(f"    High mask crops (>10% mask): {len(high_mask)} crops, avg Dice: {high_mask_dice:.3f}")

        if low_mask:
            low_mask_dice = np.mean([c['dice'] for c in low_mask])
            print(f"    Low mask crops (≤1% mask): {len(low_mask)} crops, avg Dice: {low_mask_dice:.3f}")

    # Normalize by weights (avoid division by zero)
    final_pred = np.divide(pred_sum, weight_sum, out=np.zeros_like(pred_sum), where=weight_sum!=0)

    # Apply threshold and crop to original size
    final_mask = (final_pred > threshold).astype(np.uint8) * 255
    final_mask = final_mask[:original_h, :original_w]

    if compute_crop_dice:
        return final_mask, crop_dice_scores
    else:
        return final_mask

def predict_large_image_memory_efficient(image, model, device, crop_size=512, stride=128,
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

def validate_model(model, val_loader, loss_fn, device):
    """Validation function"""
    model.eval()
    val_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).unsqueeze(1).float()

            # Forward pass
            preds = model(images)
            loss = loss_fn(preds, masks)
            val_loss += loss.item()

            # Calculate Dice score
            preds_sigmoid = torch.sigmoid(preds)
            preds_binary = (preds_sigmoid > 0.5).float()

            intersection = (preds_binary * masks).sum()
            union = preds_binary.sum() + masks.sum()
            dice = (2. * intersection) / (union + 1e-7)
            dice_scores.append(dice.item())

    avg_val_loss = val_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)

    return avg_val_loss, avg_dice

def plot_training_history(train_losses, val_losses, train_dice_scores, val_dice_scores):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Dice scores
    ax2.plot(train_dice_scores, label='Train Dice', color='blue')
    ax2.plot(val_dice_scores, label='Val Dice', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()




train_directory = ""
test_directory = ""
if len(sys.argv) != 3:
    print("Usage: python teeth_model.py train_directory test_directory")
    sys.exit(1)
else:
    train_directory = sys.argv[1]
    test_directory = sys.argv[2]





# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Test GPU functionality
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    test_tensor = torch.randn(1000, 1000).cuda()
    result = torch.matmul(test_tensor, test_tensor)
    print("✓ GPU computation test passed")
    del test_tensor, result
else:
    print("GPU not functional")
    sys.exit(1)
# Model setup
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # can be 'efficientnet-b0', etc.
    encoder_weights="imagenet",     # use "None" if training from scratch
    in_channels=3,
    classes=1,                      # binary segmentation
)
model = model.to(device)


# Simple transform for training (since data is already augmented through cropping)
train_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2(),
], additional_targets={"mask": "mask"})

# Validation transform (no augmentation)
val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2(),
], additional_targets={"mask": "mask"})

# Create full dataset
full_dataset = SegmentationDataset(os.path.join(train_directory, "images"),
                                   os.path.join(train_directory, "masks"),
                                   train_transform)

# Split dataset into train and validation (80/20 split)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
print(f"Dataset split: {train_size} training, {val_size} validation")

# Create train and validation datasets
train_dataset, val_indices = random_split(full_dataset, [train_size, val_size])

# Create validation dataset with different transform
val_dataset = SegmentationDataset(os.path.join(train_directory, "images"),
                                  os.path.join(train_directory, "masks"),
                                  val_transform)

val_dataset.images = [full_dataset.images[i] for i in val_indices.indices]

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                        num_workers=2, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)
# Move model to GPU
model = model.to(device)
loss_fn = smp.losses.DiceLoss("binary")
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Added weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training setup
num_epochs = 25
best_val_dice = 0.0
patience = 25
patience_counter = 0
early_stop = False

# For plotting
train_losses = []
val_losses = []
train_dice_scores = []
val_dice_scores = []

print("Starting training with validation...")
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_dist():
    # SLURM or torchrun exports these:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    return rank, local_rank, world_size

rank, local_rank, world_size = setup_dist()
torch.cuda.set_device(local_rank)

device = torch.device(f"cuda:{local_rank}")
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
per_gpu_batch = 4
# Use a DistributedSampler for the training set:
from torch.utils.data.distributed import DistributedSampler
train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler,
                          num_workers=2, pin_memory=True, persistent_workers=True)

# For validation you can keep a regular DataLoader, or use another DistributedSampler and
# compute metrics on rank 0 only. Example: only save/print on rank 0:
is_main = (rank == 0)
if is_main:
    torch.save(model.module.state_dict(), "best_model.pth")  # note model.module under DDP

# Use a DistributedSampler for training
train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
train_loader = DataLoader(train_dataset, batch_size=4,
                          sampler=train_sampler, num_workers=2, pin_memory=True,
                          persistent_workers=True)
# remember each epoch:
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    if early_stop:
        print("Early stopping triggered!")
        break

    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Training phase
    model.train()
    train_loss = 0.0
    train_dice_sum = 0.0
    batch_count = 0

    for images, masks in train_loader:
        # Move data to GPU
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1).float()

        # Forward pass
        preds = model(images)
        loss = loss_fn(preds, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        train_loss += loss.item()
        batch_count += 1

        # Calculate training Dice score
        with torch.no_grad():
            preds_sigmoid = torch.sigmoid(preds)
            preds_binary = (preds_sigmoid > 0.5).float()
            intersection = (preds_binary * masks).sum()
            union = preds_binary.sum() + masks.sum()
            dice = (2. * intersection) / (union + 1e-7)
            train_dice_sum += dice.item()

        if batch_count % 10 == 0:
            print(f"  Batch {batch_count}, Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")

    # Calculate average training metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice_sum / len(train_loader)

    # Validation phase
    avg_val_loss, avg_val_dice = validate_model(model, val_loader, loss_fn, device)

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Store metrics for plotting
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_dice_scores.append(avg_train_dice)
    val_dice_scores.append(avg_val_dice)

    # Print epoch summary
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}, Val Dice:   {avg_val_dice:.4f}")
    print(f"  Loss Gap:   {avg_val_loss - avg_train_loss:.4f}")

    # Check for overfitting
    loss_gap = avg_val_loss - avg_train_loss
    if loss_gap > 0.3:
        print("  ⚠️  OVERFITTING DETECTED! (Loss gap > 0.3)")
    elif loss_gap > 0.15:
        print("  ⚠️  Possible overfitting (Loss gap > 0.15)")

    # Early stopping and model saving
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ New best validation Dice: {best_val_dice:.4f} - Model saved!")
    else:
        patience_counter += 1
        print(f"  No improvement for {patience_counter} epochs")

        if patience_counter >= patience:
            early_stop = True

print("\nTraining completed!")
print(f"Best validation Dice score: {best_val_dice:.4f}")

# Plot training history
plot_training_history(train_losses, val_losses, train_dice_scores, val_dice_scores)

# Load best model for testing
print("\nLoading best model for testing...")
model.load_state_dict(torch.load("best_model.pth"))

# Testing/Evaluation - LOAD TEST IMAGES AT ORIGINAL SIZE
print("Starting evaluation on test set...")

# Create test dataset class that preserves original dimensions
class TestDatasetOriginalSize(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Load images at their ORIGINAL size - no resizing!
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")
        return image, mask, img_name

# Create dataset but iterate directly (no DataLoader to avoid PIL Image batching issues)
test_dataset = TestDatasetOriginalSize(os.path.join(test_directory, "images"),
                                       os.path.join(test_directory, "masks"))

print(f"Test dataset created with {len(test_dataset)} images (original sizes preserved)")

# Set model to eval mode
model.eval()

# Optional: make a folder to save predicted masks
os.makedirs("predictions", exist_ok=True)

# For evaluation
dice_scores = []

with torch.no_grad():
    # Iterate directly through dataset indices instead of using DataLoader
    for i in range(len(test_dataset)):
        pil_image, pil_mask, img_name = test_dataset[i]

        print(f"\nProcessing test image {i+1}/{len(test_dataset)}: {img_name}")
        print(f"ORIGINAL image size: {pil_image.size}")  # This should show the real size now!

        # Choose prediction method based on image size
        image_area = pil_image.size[0] * pil_image.size[1]

        print("Using standard prediction with TTA and crop-level Dice tracking")
        result = predict_large_image(
                pil_image, model, device,
                crop_size=512, stride=256, batch_size=4, threshold=0.5,
                use_tta=True, use_gaussian_weights=True,
                compute_crop_dice=True, ground_truth_mask=pil_mask
            )

            # Handle return value (either just pred or (pred, crop_dice_scores))
        if isinstance(result, tuple):
                pred, crop_dice_scores = result
        else:
            pred = result
            crop_dice_scores = None

        print(f"Final prediction size: {pred.shape[1]}×{pred.shape[0]}")

        # Convert prediction to tensor
        pred_tensor = torch.from_numpy(pred / 255.0).unsqueeze(0).float()  # [1, H, W]

        # Save prediction
        save_image(pred_tensor, f"predictions/pred_{i}_{img_name}")

        # Convert ground truth mask to tensor at original size
        mask_np = np.array(pil_mask)
        mask_tensor = torch.from_numpy((mask_np > 127).astype('float32')).unsqueeze(0).float()
        mask_tensor = mask_tensor.to(device)
        pred_tensor = pred_tensor.to(device)

        # Verify sizes match (they should now!)
        if mask_tensor.shape[-2:] != pred_tensor.shape[-2:]:
            print(f"⚠️  WARNING: Size mismatch! Mask: {mask_tensor.shape[-2:]}, Pred: {pred_tensor.shape[-2:]}")
            print("This shouldn't happen with the new approach!")
            # Resize as fallback
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0), size=pred_tensor.shape[-2:], mode='nearest'
            ).squeeze(0)

        # Compute Dice score
        intersection = (pred_tensor * mask_tensor).sum()
        union = pred_tensor.sum() + mask_tensor.sum()
        dice = (2. * intersection) / (union + 1e-7)
        dice_scores.append(dice.item())

        print(f"Test image {i}: Dice score = {dice.item():.4f}")

# Summary
mean_dice = np.mean(dice_scores)
print(f"\nFinal Results:")
print(f"Best validation Dice score: {best_val_dice:.4f}")
print(f"Average test Dice score: {mean_dice:.4f}")
print(f"Model saved as 'best_model.pth'")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU cache cleared.")

print("\n=== TRAINING ANALYSIS ===")
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

print("\n=== PREDICTION USAGE EXAMPLES ===")
print("For production (fast, good quality):")
print("pred = predict_large_image(image, model, device, use_tta=False, use_gaussian_weights=True)")
print("\nFor best quality (slower):")
print("pred = predict_large_image(image, model, device, use_tta=True, stride=96)")
print("\nFor very large images:")
print("pred = predict_large_image_memory_efficient(image, model, device, max_memory_gb=4)")
dist.destroy_process_group()
