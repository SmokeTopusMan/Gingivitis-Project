# test_only_teeth_model.py
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# -----------------------------
# Prediction utilities
# -----------------------------
def create_gaussian_weight_map(crop_size, sigma=None):
    """Create a 2D Gaussian weight map that gives higher weights to center pixels."""
    if sigma is None:
        sigma = crop_size / 6  # Adjust this for more/less aggressive weighting
    x = np.arange(crop_size)
    y = np.arange(crop_size)
    x, y = np.meshgrid(x, y)
    cx, cy = crop_size // 2, crop_size // 2
    weight_map = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return weight_map.astype(np.float32)

def reverse_tta_predictions(predictions, transform_name):
    """Reverse TTA transformations on predictions (predictions: [B,1,H,W])."""
    if 'hflip' in transform_name:
        predictions = torch.flip(predictions, dims=[3])  # Flip width
    if 'vflip' in transform_name:
        predictions = torch.flip(predictions, dims=[2])  # Flip height
    return predictions

def process_crop_batch(crops, model, device, transform, transform_name='base'):
    """Process a batch of crops through the model; returns numpy probs [B,H,W]."""
    batch_tensors = []
    for crop in crops:
        augmented = transform(image=crop, mask=np.zeros((crop.shape[0], crop.shape[1])))
        batch_tensors.append(augmented["image"])
    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        predictions = model(batch_tensor)
        predictions = torch.sigmoid(predictions)
        if transform_name != 'base':
            predictions = reverse_tta_predictions(predictions, transform_name)
        predictions = predictions.squeeze(1).cpu().numpy()

    return predictions

def predict_large_image(image, model, device, crop_size=512, stride=256,
                        batch_size=4, threshold=0.5, use_tta=True,
                        use_gaussian_weights=True, compute_crop_dice=False,
                        ground_truth_mask=None):
    """
    Sliding-window prediction with batch processing, optional TTA, and Gaussian blending.
    If compute_crop_dice=True and ground_truth_mask provided, returns (final_mask, crop_stats).
    Otherwise returns final_mask.
    """
    image_np = np.array(image)
    original_h, original_w = image_np.shape[:2]

    # Padding to make sliding windows align nicely
    pad_h = (stride - (original_h - crop_size) % stride) % stride if original_h > crop_size else max(0, crop_size - original_h)
    pad_w = (stride - (original_w - crop_size) % stride) % stride if original_w > crop_size else max(0, crop_size - original_w)
    padded_img = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    padded_h, padded_w = padded_img.shape[:2]

    # Optional GT padding for per-crop Dice
    padded_gt_mask = None
    if compute_crop_dice and ground_truth_mask is not None:
        gt_mask_np = np.array(ground_truth_mask)
        padded_gt_mask = np.pad(gt_mask_np, ((0, pad_h), (0, pad_w)), mode='reflect')

    pred_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_map = create_gaussian_weight_map(crop_size) if use_gaussian_weights else np.ones((crop_size, crop_size), dtype=np.float32)

    base_transform = A.Compose([
        A.Resize(crop_size, crop_size),
        A.Normalize(),
        ToTensorV2(),
    ])

    tta_transforms = {}
    if use_tta:
        tta_transforms = {
            'hflip': A.Compose([A.HorizontalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
            'vflip': A.Compose([A.VerticalFlip(p=1.0),   A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
            'hvflip': A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
        }

    crop_coords = [(i, j) for i in range(0, padded_h - crop_size + 1, stride)
                          for j in range(0, padded_w - crop_size + 1, stride)]
    print(f"Processing {len(crop_coords)} crops in batches of {batch_size}")

    crop_dice_scores = []
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(crop_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(crop_coords))
            batch_coords = crop_coords[batch_start:batch_end]

            batch_crops, batch_gt_crops = [], []
            for (i, j) in batch_coords:
                crop = padded_img[i:i+crop_size, j:j+crop_size, :]
                batch_crops.append(crop)
                if compute_crop_dice and padded_gt_mask is not None:
                    gt_crop = padded_gt_mask[i:i+crop_size, j:j+crop_size]
                    batch_gt_crops.append(gt_crop)

            # Base predictions
            batch_predictions = process_crop_batch(batch_crops, model, device, base_transform, 'base')

            # TTA predictions (average)
            if use_tta:
                tta_predictions = []
                for tta_name, tta_transform in tta_transforms.items():
                    tta_pred = process_crop_batch(batch_crops, model, device, tta_transform, tta_name)
                    tta_predictions.append(tta_pred)
                all_predictions = [batch_predictions] + tta_predictions
                batch_predictions = np.mean(all_predictions, axis=0)

            # Optional per-crop Dice
            if compute_crop_dice and batch_gt_crops:
                for idx, (i, j) in enumerate(batch_coords):
                    pred_crop = batch_predictions[idx]
                    gt_crop = batch_gt_crops[idx]
                    pred_binary = (pred_crop > threshold).astype(np.float32)
                    gt_binary = (gt_crop > 127).astype(np.float32)
                    intersection = (pred_binary * gt_binary).sum()
                    union = pred_binary.sum() + gt_binary.sum()
                    dice = (2. * intersection) / (union + 1e-7)
                    crop_dice_scores.append({
                        'batch': batch_start // batch_size,
                        'crop_idx': batch_start + idx,
                        'position': (i, j),
                        'dice': float(dice),
                        'mask_coverage': float(gt_binary.mean()),
                        'pred_coverage': float(pred_binary.mean()),
                    })

            # Blend into canvas
            for idx, (i, j) in enumerate(batch_coords):
                pred = batch_predictions[idx]
                pred_sum[i:i+crop_size, j:j+crop_size] += pred * weight_map
                weight_sum[i:i+crop_size, j:j+crop_size] += weight_map

            # Progress log
            if ((batch_start // batch_size) + 1) % 10 == 0:
                progress = (batch_end / len(crop_coords)) * 100
                if compute_crop_dice and crop_dice_scores:
                    recent = [c['dice'] for c in crop_dice_scores[-len(batch_coords):]]
                    print(f"  Progress: {progress:.1f}%, Recent crop Dice: {np.mean(recent):.3f}")
                else:
                    print(f"  Progress: {progress:.1f}%")

    # Normalize and trim to original size
    final_pred = np.divide(pred_sum, weight_sum, out=np.zeros_like(pred_sum), where=weight_sum != 0)
    final_mask = (final_pred > threshold).astype(np.uint8) * 255
    final_mask = final_mask[:original_h, :original_w]

    if compute_crop_dice:
        return final_mask, crop_dice_scores
    return final_mask

def predict_large_image_memory_efficient(image, model, device, crop_size=512, stride=256,
                                         threshold=0.5, max_memory_gb=4,
                                         compute_crop_dice=False):
    """
    Memory-efficient version for very large images: processes vertical chunks.
    """
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    bytes_per_pixel = 4  # float32 estimate
    estimated_memory = (h * w * bytes_per_pixel) / (1024**3)  # GB
    print(f"Estimated memory usage: {estimated_memory:.2f} GB")

    if estimated_memory <= max_memory_gb:
        print("Using standard processing")
        return predict_large_image(image, model, device, crop_size, stride,
                                   batch_size=4, threshold=threshold,
                                   use_tta=True, use_gaussian_weights=True,
                                   compute_crop_dice=compute_crop_dice,
                                   ground_truth_mask=None)

    print("Using memory-efficient processing (chunks)")
    chunk_height = int((max_memory_gb * 1024**3) / (w * bytes_per_pixel))
    chunk_height = max(chunk_height, crop_size * 2)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    start = 0
    while start < h:
        end = min(start + chunk_height, h)
        print(f"Processing chunk: {start}-{end} ({end-start}x{w})")
        chunk = image_np[start:end, :, :]
        chunk_pil = Image.fromarray(chunk)
        chunk_mask = predict_large_image(
            chunk_pil, model, device, crop_size, stride,
            batch_size=4, threshold=threshold,
            use_tta=True, use_gaussian_weights=True
        )

        if start > 0:
            overlap = crop_size
            blend_start = max(0, crop_size // 2)
            # Blend overlap band
            for i in range(blend_start):
                alpha = i / blend_start
                final_mask[start + i, :] = (
                    alpha * chunk_mask[i, :] + (1 - alpha) * final_mask[start + i, :]
                ).astype(np.uint8)
            final_mask[start + blend_start:end, :] = chunk_mask[blend_start:, :]
        else:
            final_mask[start:end, :] = chunk_mask
        start = end

    return final_mask

# -----------------------------
# Minimal test dataset (original sizes)
# -----------------------------
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

# -----------------------------
# Main: load model, run evaluation
# -----------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python test_only_teeth_model.py <test_directory>")
        sys.exit(1)

    test_directory  = sys.argv[1]  # <-- fixed: was sys.argv[1] before

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("GPU not functional (CUDA not available). Exiting.")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model (must match the architecture used for training)
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # weights will come from the checkpoint
        in_channels=3,
        classes=1,
    ).to(device)

    # Load best weights
    ckpt_path = os.path.join(os.getcwd(), "best_model.pth")
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    state_dict = torch.load(ckpt_path, map_location=device)
    # Remove 'module.' prefix from all keys
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded checkpoint: best_model.pth")

    # Test dataset
    test_dataset = TestDatasetOriginalSize(
        os.path.join(test_directory, "images"),
        os.path.join(test_directory, "masks")
    )
    print(f"Test dataset created with {len(test_dataset)} images (original sizes preserved)")

    # Output dir
    os.makedirs("predictions", exist_ok=True)

    # Evaluate
    dice_scores = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            pil_image, pil_mask, img_name = test_dataset[i]
            print(f"\nProcessing test image {i+1}/{len(test_dataset)}: {img_name}")
            print(f"ORIGINAL image size: {pil_image.size}")

            # Standard high-quality prediction: TTA + Gaussian blending
            result = predict_large_image(
                pil_image, model, device,
                crop_size=512, stride=256, batch_size=4, threshold=0.5,
                use_tta=True, use_gaussian_weights=True,
                compute_crop_dice=True, ground_truth_mask=pil_mask
            )

            if isinstance(result, tuple):
                pred, crop_dice_scores = result
            else:
                pred = result
                crop_dice_scores = None

            print(f"Final prediction size: {pred.shape[1]}×{pred.shape[0]}")

            # Save prediction
            pred_tensor = torch.from_numpy(pred / 255.0).unsqueeze(0).float()  # [1,H,W]
            save_image(pred_tensor, f"predictions/pred_{i}_{img_name}")

            # Dice vs GT (at original size)
            mask_np = np.array(pil_mask)
            mask_tensor = torch.from_numpy((mask_np > 127).astype('float32')).unsqueeze(0).float().to(device)
            pred_tensor = pred_tensor.to(device)

            if mask_tensor.shape[-2:] != pred_tensor.shape[-2:]:
                # Fallback (shouldn't happen)
                print(f"⚠️  WARNING: Size mismatch! Mask: {mask_tensor.shape[-2:]}, Pred: {pred_tensor.shape[-2:]}")
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0), size=pred_tensor.shape[-2:], mode='nearest'
                ).squeeze(0)

            intersection = (pred_tensor * mask_tensor).sum()
            union = pred_tensor.sum() + mask_tensor.sum()
            dice = (2. * intersection) / (union + 1e-7)
            dice_scores.append(dice.item())
            print(f"Test image {i}: Dice score = {dice.item():.4f}")

    mean_dice = np.mean(dice_scores) if dice_scores else float('nan')
    print("\nFinal Results:")
    print(f"Average test Dice score: {mean_dice:.4f}")
    print("Predictions saved to ./predictions")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

if __name__ == "__main__":
    main()
