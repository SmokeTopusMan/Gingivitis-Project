import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from PIL import Image
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# -------------------------
# DDP setup (same spirit as yours)
# -------------------------
def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1

    if not is_dist:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, local_rank, world_size, device

IS_DIST, RANK, LOCAL_RANK, WORLD_SIZE, DEVICE = setup_dist()
IS_MAIN = (RANK == 0)

# -------------------------
# Helpers copied/adapted from your code
# -------------------------
def is_pure_black_rgb(crop_rgb, thr=3, min_fraction=0.999):
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        return False
    black_pixels = np.all(crop_rgb <= thr, axis=2)
    return black_pixels.mean() >= min_fraction

def create_gaussian_weight_map(crop_size, sigma=None):
    if sigma is None:
        sigma = crop_size / 6
    x = np.arange(crop_size)
    y = np.arange(crop_size)
    x, y = np.meshgrid(x, y)
    cx, cy = crop_size // 2, crop_size // 2
    weight_map = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return weight_map.astype(np.float32)

def reverse_tta_predictions(predictions, transform_name):
    if 'hflip' in transform_name:
        predictions = torch.flip(predictions, dims=[3])
    if 'vflip' in transform_name:
        predictions = torch.flip(predictions, dims=[2])
    return predictions

def process_crop_batch(crops, model, device, transform, transform_name='base'):
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

def predict_large_image_probs(
    image, model, device,
    crop_size=256, stride=128, batch_size=4,
    use_tta=True, use_gaussian_weights=True,
    black_thr=3, black_min_fraction=0.999, force_black_pred=True
):
    """
    Same as your predict_large_image(), BUT returns float probability map
    (H x W, values in [0,1]) BEFORE thresholding.
    """
    image_np = np.array(image)
    original_h, original_w = image_np.shape[:2]

    pad_h = (stride - (original_h - crop_size) % stride) % stride if original_h > crop_size else max(0, crop_size - original_h)
    pad_w = (stride - (original_w - crop_size) % stride) % stride if original_w > crop_size else max(0, crop_size - original_w)

    padded_img = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    padded_h, padded_w = padded_img.shape[:2]

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
            'vflip': A.Compose([A.VerticalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
            'hvflip': A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(), ToTensorV2()]),
        }

    crop_coords = [(i, j)
                   for i in range(0, padded_h - crop_size + 1, stride)
                   for j in range(0, padded_w - crop_size + 1, stride)]

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(crop_coords), batch_size):
            batch_end = min(batch_start + batch_size, len(crop_coords))
            batch_coords = crop_coords[batch_start:batch_end]

            batch_crops = []
            batch_is_black = []

            for i, j in batch_coords:
                crop = padded_img[i:i+crop_size, j:j+crop_size, :]
                batch_crops.append(crop)
                batch_is_black.append(is_pure_black_rgb(crop, thr=black_thr, min_fraction=black_min_fraction))

            # Base preds
            batch_predictions = process_crop_batch(batch_crops, model, device, base_transform, 'base')

            # TTA preds
            if use_tta:
                tta_preds = []
                for tta_name, tta_transform in tta_transforms.items():
                    tta_pred = process_crop_batch(batch_crops, model, device, tta_transform, tta_name)
                    tta_preds.append(tta_pred)
                batch_predictions = np.mean([batch_predictions] + tta_preds, axis=0)

            # Force black crops to 0
            if force_black_pred:
                for k, is_blk in enumerate(batch_is_black):
                    if is_blk:
                        batch_predictions[k].fill(0.0)

            # Accumulate
            for idx, (i, j) in enumerate(batch_coords):
                pred = batch_predictions[idx]
                pred_sum[i:i+crop_size, j:j+crop_size] += pred * weight_map
                weight_sum[i:i+crop_size, j:j+crop_size] += weight_map

    final_pred = np.divide(pred_sum, weight_sum, out=np.zeros_like(pred_sum), where=weight_sum != 0)
    final_pred = final_pred[:original_h, :original_w]  # crop back to original size

    orig_np = np.array(image)  # original RGB
    pure_black_mask = np.all(orig_np <= 3, axis=2)  # same threshold as your function uses
    final_pred[pure_black_mask] = 0.0

    return final_pred 

def dice_from_binary(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    pred_bin, gt_bin: {0,1} arrays same shape.
    Dice=1 if both empty.
    """
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    if union == 0:
        return 1.0
    return float((2.0 * inter) / (union + 1e-7))

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

def load_checkpoint_safely(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    # unwrap common nesting conventions
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"{path} is not a dict-like checkpoint")

    # keep only tensors
    sd = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    if not sd:
        raise RuntimeError(f"No tensor weights found in {path}")

    model_has_module = any(k.startswith("module.") for k in model.state_dict().keys())
    ckpt_has_module  = any(k.startswith("module.") for k in sd.keys())

    # FORCE prefix alignment
    if ckpt_has_module and not model_has_module:
        consume_prefix_in_state_dict_if_present(sd, "module.")
    elif (not ckpt_has_module) and model_has_module:
        sd = {"module." + k: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"[load_checkpoint_safely] Missing keys ({len(missing)}): {missing[:10]} ...")
    if unexpected:
        print(f"[load_checkpoint_safely] Unexpected keys ({len(unexpected)}): {unexpected[:10]} ...")

    print(f"[load_checkpoint_safely] Loaded {path} successfully.")

# -------------------------
# Test dataset (original size)
# -------------------------
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
        mask  = Image.open(os.path.join(self.masks_dir, img_name)).convert("L")
        return image, mask, img_name

# -------------------------
# Main
# -------------------------
def main():
    # --- Model setup (same as yours) ---
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    if IS_DIST:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    if IS_MAIN:
        print("Loading best model for threshold search...")
    core_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    load_checkpoint_safely(core_model, "best_model_ging_clean.pth")

    # --- Test data ---
    test_images_dir = "downloads/test/relevant_images"
    test_masks_dir  = "downloads/test/masks"
    test_dataset = TestDatasetOriginalSize(test_images_dir, test_masks_dir)

    n_test = len(test_dataset)
    if IS_MAIN:
        print(f"Test dataset: {n_test} images (original sizes).")
        os.makedirs("predictions", exist_ok=True)

    # Optional split per-rank for speed
    indices = list(range(n_test))
    if IS_DIST:
        indices = indices[RANK::WORLD_SIZE]

    # ---- Threshold sweep locally, reduce only scalars ----
    thresholds = np.arange(0.05, 0.96, 0.05)  # 0.05,0.10,...0.95
    n_thr = len(thresholds)

    sum_dice_all = np.zeros(n_thr, dtype=np.float64)
    sum_dice_pos = np.zeros(n_thr, dtype=np.float64)
    count_all    = np.zeros(n_thr, dtype=np.int64)
    count_pos    = np.zeros(n_thr, dtype=np.int64)

    if IS_MAIN:
        print("\nSweeping thresholds (distributed, no prob gather)...")

    for k, i in enumerate(indices):
        pil_image, pil_mask, img_name = test_dataset[i]
        if IS_MAIN:
            print(f"[rank{RANK}] probs for {img_name} ({k+1}/{len(indices)})")

        prob_map = predict_large_image_probs(
            pil_image, model, DEVICE,
            crop_size=256, stride=128, batch_size=4,
            use_tta=True, use_gaussian_weights=True,
            black_thr=3, black_min_fraction=0.999, force_black_pred=True
        )

        gt_np = (np.array(pil_mask) > 127).astype(np.uint8)
        gt_has_pos = gt_np.sum() > 0

        for t_idx, thr in enumerate(thresholds):
            pred_bin = (prob_map > thr).astype(np.uint8)
            d = dice_from_binary(pred_bin, gt_np)

            sum_dice_all[t_idx] += d
            count_all[t_idx]    += 1
            if gt_has_pos:
                sum_dice_pos[t_idx] += d
                count_pos[t_idx]    += 1

        # free per-image memory asap
        del prob_map

    # Reduce across ranks (tiny tensors)
    if IS_DIST and dist.is_initialized() and WORLD_SIZE > 1:
        t_all = torch.tensor(sum_dice_all, device=DEVICE, dtype=torch.float64)
        t_pos = torch.tensor(sum_dice_pos, device=DEVICE, dtype=torch.float64)
        c_all = torch.tensor(count_all, device=DEVICE, dtype=torch.int64)
        c_pos = torch.tensor(count_pos, device=DEVICE, dtype=torch.int64)

        dist.all_reduce(t_all, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(c_all, op=dist.ReduceOp.SUM)
        dist.all_reduce(c_pos, op=dist.ReduceOp.SUM)

        sum_dice_all = t_all.cpu().numpy()
        sum_dice_pos = t_pos.cpu().numpy()
        count_all    = c_all.cpu().numpy()
        count_pos    = c_pos.cpu().numpy()

    if not IS_MAIN:
        return  # only rank0 picks best threshold & saves masks

    mean_all = sum_dice_all / np.maximum(count_all, 1)
    mean_pos = sum_dice_pos / np.maximum(count_pos, 1)

    best_idx = int(np.argmax(mean_all))
    best_thr = float(thresholds[best_idx])

    for thr, ma, mp, cp in zip(thresholds, mean_all, mean_pos, count_pos):
        print(f"  thr={thr:.2f}  mean_dice_all={ma:.4f}  mean_dice_pos={mp:.4f} (pos n={cp})")

    print("\n=== BEST THRESHOLD FOUND ===")
    print(f"Best threshold: {best_thr:.2f}")
    print(f"Best mean Dice (all images): {mean_all[best_idx]:.4f}")
    print(f"Mean Dice (positive-only) at best thr: {mean_pos[best_idx]:.4f}")

    # ---- Save predictions for best threshold ----
    # We recompute prob maps on rank0 only (no big storage / gather)
    print("\nSaving best-threshold predictions to predictions/ ...")
    for i in range(n_test):
        pil_image, _, img_name = test_dataset[i]

        prob_map = predict_large_image_probs(
            pil_image, model, DEVICE,
            crop_size=256, stride=128, batch_size=4,
            use_tta=True, use_gaussian_weights=True,
            black_thr=3, black_min_fraction=0.999, force_black_pred=True
        )

        pred_mask = (prob_map > best_thr).astype(np.uint8) * 255
        pred_tensor = torch.from_numpy(pred_mask / 255.0).unsqueeze(0).float()  # [1,H,W]

        save_image(pred_tensor, f"predictions/pred_{i}_{img_name}")

        del prob_map

    print("Done.")

if __name__ == "__main__":
    main()
    if IS_DIST and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
