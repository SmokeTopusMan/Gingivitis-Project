# tools/postprocess_black_pixels.py
import os
import argparse
import numpy as np
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
MASK_EXTS = (".png",)

def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)

def is_mask_file(name: str) -> bool:
    return name.lower().endswith(MASK_EXTS)

def main():
    parser = argparse.ArgumentParser(description="Postprocess masks: zero-out mask pixels where the original image is black.")
    parser.add_argument("--images_dir", required=True, help="Directory with the ORIGINAL images.")
    parser.add_argument("--masks_dir", required=True, help="Directory with predicted masks (png).")
    parser.add_argument("--out_dir", required=True, help="Where to write postprocessed masks.")
    parser.add_argument("--thr", type=int, default=3, help="Black threshold: pixel is black if all RGB <= thr.")
    args = parser.parse_args()

    images_dir = args.images_dir
    masks_dir = args.masks_dir
    out_dir = args.out_dir
    thr = args.thr

    if not os.path.isdir(images_dir):
        raise SystemExit(f"[ERROR] images_dir not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise SystemExit(f"[ERROR] masks_dir not found: {masks_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # We only need to process masks that exist.
    mask_files = [f for f in os.listdir(masks_dir) if is_mask_file(f)]
    if not mask_files:
        print(f"[WARN] No masks found in {masks_dir}. Nothing to postprocess.")
        return

    processed = 0
    skipped = 0

    for mask_name in mask_files:
        stem = os.path.splitext(mask_name)[0]

        # Find a matching image by trying common extensions
        img_path = None
        for ext in IMG_EXTS:
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        # If your masks are "name.png" while originals are "name.jpg", this will still find it.
        if img_path is None:
            # fallback: try exact name match (in case original is png and same file name)
            candidate = os.path.join(images_dir, mask_name)
            if os.path.exists(candidate) and is_image_file(candidate):
                img_path = candidate

        if img_path is None:
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(os.path.join(masks_dir, mask_name)).convert("L")

            # If sizes differ, align mask to image (your overlay code already does this too)
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)

            img_np = np.array(img)  # HxWx3
            mask_np = np.array(mask)  # HxW (0..255)

            black_pixels = np.all(img_np <= thr, axis=2)  # HxW boolean
            mask_np[black_pixels] = 0

            out_path = os.path.join(out_dir, mask_name)
            Image.fromarray(mask_np).save(out_path)

            processed += 1

        except Exception as e:
            print(f"[WARN] Failed processing {mask_name}: {e}")
            skipped += 1

    print(f"[INFO] Postprocess done. processed={processed}, skipped={skipped}")
    print(f"[INFO] Output masks: {out_dir}")

if __name__ == "__main__":
    main()

