import os
import sys
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import argparse


def create_gaussian_weight_map(crop_size, sigma=None):
    if sigma is None:
        sigma = crop_size / 6
    x = np.arange(crop_size)
    y = np.arange(crop_size)
    xx, yy = np.meshgrid(x, y)
    cx = cy = crop_size // 2
    weight_map = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
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
        aug = transform(image=crop, mask=np.zeros((crop.shape[0], crop.shape[1])))
        batch_tensors.append(aug["image"])
    batch_tensor = torch.stack(batch_tensors).to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(batch_tensor)
        preds = torch.sigmoid(logits)
        if transform_name != 'base':
            preds = reverse_tta_predictions(preds, transform_name)
        preds = preds.squeeze(1).cpu().numpy()
    return preds


def predict_large_image(
        image_pil,
        model,
        device,
        crop_size=512,
        stride=256,
        batch_size=4,
        threshold=0.5,
        use_tta=False,
        use_gaussian_weights=True
):
    image_np = np.array(image_pil.convert("RGB"))
    H, W = image_np.shape[:2]

    pad_h = (stride - (H - crop_size) % stride) % stride if H > crop_size else max(0, crop_size - H)
    pad_w = (stride - (W - crop_size) % stride) % stride if W > crop_size else max(0, crop_size - W)
    padded = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    PH, PW = padded.shape[:2]

    pred_sum = np.zeros((PH, PW), dtype=np.float32)
    weight_sum = np.zeros((PH, PW), dtype=np.float32)
    weight_map = create_gaussian_weight_map(crop_size) if use_gaussian_weights else np.ones((crop_size, crop_size),
                                                                                            dtype=np.float32)

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
            'hvflip': A.Compose(
                [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Resize(crop_size, crop_size), A.Normalize(),
                 ToTensorV2()]),
        }

    coords = [(i, j) for i in range(0, PH - crop_size + 1, stride)
              for j in range(0, PW - crop_size + 1, stride)]

    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]
            batch_crops = [padded[i:i + crop_size, j:j + crop_size, :] for (i, j) in batch_coords]

            base_pred = process_crop_batch(batch_crops, model, device, base_transform, 'base')

            if use_tta:
                tta_list = []
                for tname, ttrans in tta_transforms.items():
                    tta_list.append(process_crop_batch(batch_crops, model, device, ttrans, tname))
                all_preds = [base_pred] + tta_list
                batch_pred = np.mean(all_preds, axis=0)
            else:
                batch_pred = base_pred

            for k, (i, j) in enumerate(batch_coords):
                pred_k = batch_pred[k]
                pred_sum[i:i + crop_size, j:j + crop_size] += pred_k * weight_map
                weight_sum[i:i + crop_size, j:j + crop_size] += weight_map

    final_pred = np.divide(pred_sum, weight_sum, out=np.zeros_like(pred_sum), where=weight_sum != 0)
    final_mask = (final_pred > threshold).astype(np.uint8) * 255
    return final_mask[:H, :W]


def load_model(weights_path, device, encoder_name="resnet34", encoder_weights="imagenet"):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)
    state = torch.load(weights_path, map_location=device)

    if list(state.keys())[0].startswith('module.'):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    model.load_state_dict(state)
    return model


def is_image_file(name):
    return name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))


def main():
    parser = argparse.ArgumentParser(
        description="UNet++ General Segmentation - Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", required=True, help="Path to trained model weights file (.pth)")
    parser.add_argument("--input", required=True, help="Directory containing input images")
    parser.add_argument("--output", required=True, help="Directory where predicted masks will be saved")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Encoder architecture (default: resnet34)")
    parser.add_argument("--encoder-weights", type=str, default="imagenet",
                        help="Encoder pretrained weights (default: imagenet)")
    parser.add_argument("--crop-size", type=int, default=512, help="Size of sliding window crops (default: 512)")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for sliding window (default: 256)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of crops to process simultaneously (default: 4)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary mask (default: 0.5)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation")
    parser.add_argument("--no-gauss", action="store_true",
                        help="Disable Gaussian blending of overlapping crops")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    encoder_weights_arg = None if args.encoder_weights.lower() == 'none' else args.encoder_weights

    print(
        f"Loading model with encoder: {args.encoder}, pretrained on: {encoder_weights_arg or 'random initialization'}")
    model = load_model(args.weights, device, encoder_name=args.encoder, encoder_weights=encoder_weights_arg)
    model.eval()

    files = [f for f in sorted(os.listdir(args.input)) if is_image_file(f)]
    print(f"Found {len(files)} images in {args.input}\n")

    for idx, fname in enumerate(files, 1):
        in_path = os.path.join(args.input, fname)
        out_path = os.path.join(args.output, fname.rsplit('.', 1)[0] + ".jpg")

        try:
            pil_img = Image.open(in_path).convert("RGB")
            print(f"[{idx}/{len(files)}] {fname}  size={pil_img.size}")

            mask = predict_large_image(
                pil_img,
                model,
                device,
                crop_size=args.crop_size,
                stride=args.stride,
                batch_size=args.batch_size,
                threshold=args.threshold,
                use_tta=args.tta,
                use_gaussian_weights=not args.no_gauss
            )

            Image.fromarray(mask).save(out_path)
        except Exception as e:
            print(f"  [!] Error on {fname}: {e}")

    print(f"\nDone. Masks saved to: {args.output}")


if __name__ == "__main__":
    main()