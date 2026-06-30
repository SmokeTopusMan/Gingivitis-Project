import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def find_mask_path(masks_dir, image_filename):
    stem = os.path.splitext(os.path.basename(image_filename))[0].lower()
    for ext in IMAGE_EXTS:
        p = os.path.join(masks_dir, stem + ext)
        if os.path.isfile(p):
            return p
    for f in os.listdir(masks_dir):
        name_noext, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS and name_noext.lower() == stem:
            return os.path.join(masks_dir, f)
    return None


def dice_iou(pred_bin, gt_bin):
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return float(dice), float(iou)


def _create_final_results(images_dir, gingivitis_masks_dir, gt_masks_dir, results_dir):
    image_files = [
        f for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    print(f"\n[Stage 4] Overlaying results and computing metrics for {len(image_files)} images...")

    rows = []

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)

        # predicted gingivitis mask (from Stage 3)
        pred_mask_path = find_mask_path(gingivitis_masks_dir, img_file)
        if not pred_mask_path:
            print(f"  [!] No predicted mask for {img_file}, copying original")
            shutil.copy2(img_path, os.path.join(results_dir, img_file))
            rows.append({"image": img_file, "dice": "N/A", "iou": "N/A"})
            continue

        img = Image.open(img_path).convert("RGB")
        pred_mask = Image.open(pred_mask_path).convert("L")

        if img.size != pred_mask.size:
            pred_mask = pred_mask.resize(img.size, Image.LANCZOS)

        img_array = np.array(img)
        pred_array = np.array(pred_mask)
        _, pred_bin = cv2.threshold(pred_array, 127, 255, cv2.THRESH_BINARY)

        # green contour from predicted mask
        contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img_array.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), thickness=3)

        # compare against ground truth mask if provided
        dice, iou = "N/A", "N/A"
        if gt_masks_dir:
            gt_path = find_mask_path(gt_masks_dir, img_file)
            if gt_path:
                gt_mask = Image.open(gt_path).convert("L")
                if gt_mask.size != img.size:
                    gt_mask = gt_mask.resize(img.size, Image.LANCZOS)
                gt_array = np.array(gt_mask)
                _, gt_bin = cv2.threshold(gt_array, 127, 255, cv2.THRESH_BINARY)

                # red contour from ground truth mask
                gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, gt_contours, -1, (255, 0, 0), thickness=3)

                dice, iou = dice_iou(pred_bin, gt_bin)
                print(f"  [OK] {img_file}  Dice={dice:.4f}  IoU={iou:.4f}")
            else:
                print(f"  [!] No ground truth mask found for {img_file}")
        else:
            print(f"  [OK] {img_file}")

        Image.fromarray(result).save(os.path.join(results_dir, img_file))
        rows.append({"image": img_file, "dice": dice, "iou": iou})

    # save metrics CSV
    csv_path = os.path.join(results_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)

    numeric = [(r["dice"], r["iou"]) for r in rows if r["dice"] != "N/A"]
    if numeric:
        mean_dice = sum(d for d, _ in numeric) / len(numeric)
        mean_iou = sum(i for _, i in numeric) / len(numeric)
        print(f"\n  Mean Dice: {mean_dice:.4f}  |  Mean IoU: {mean_iou:.4f}  ({len(numeric)} images)")

    print(f"\nResults saved to: {results_dir}")
    print(f"Metrics saved to: {csv_path}")


def run(cmd, stage_name):
    print(f"\n[{stage_name}] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {stage_name} failed (exit code {e.returncode})")
        sys.exit(1)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Gingivitis detection pipeline — headless version of main_screen.py"
    )
    parser.add_argument("--images", required=True,
                        help="Directory containing input images")
    parser.add_argument("--masks", required=True,
                        help="Directory containing ground truth gingivitis masks (for comparison)")
    parser.add_argument("--results", required=True,
                        help="Directory where annotated images and metrics.csv will be saved")
    parser.add_argument("--teeth-weights",
                        default=os.path.join(project_root, "weights&results", "Teeth_model_weights.pth"),
                        help="Path to teeth segmentation model weights")
    parser.add_argument("--gingivitis-weights",
                        default=os.path.join(project_root, "weights&results", "Gingivitis_model_weights.pth"),
                        help="Path to gingivitis segmentation model weights")
    args = parser.parse_args()

    for path, name in [(args.images, "--images"),
                       (args.masks, "--masks"),
                       (args.teeth_weights, "--teeth-weights"),
                       (args.gingivitis_weights, "--gingivitis-weights")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} path does not exist: {path}")
            sys.exit(1)

    os.makedirs(args.results, exist_ok=True)

    run_model = os.path.join(project_root, "tools", "run_model.py")
    get_relevant = os.path.join(project_root, "tools", "get_relevant.py")

    temp_dir = tempfile.mkdtemp(prefix="gingivitis_lambda_")
    teeth_masks_dir = os.path.join(temp_dir, "teeth_masks")
    relevant_dir = os.path.join(temp_dir, "relevant_images")
    gingivitis_masks_dir = os.path.join(temp_dir, "gingivitis_masks")

    try:
        # Stage 1: teeth segmentation
        os.makedirs(teeth_masks_dir)
        run([sys.executable, run_model,
             "--weights", args.teeth_weights,
             "--input", args.images,
             "--output", teeth_masks_dir],
            "Stage 1 — Teeth segmentation")

        # Stage 2: crop images to relevant dental region
        # get_relevant.py writes to {argv[3]}/relevant_images/
        run([sys.executable, get_relevant,
             args.images, teeth_masks_dir, temp_dir],
            "Stage 2 — Relevant region extraction")

        # Stage 3: gingivitis segmentation on cropped images
        os.makedirs(gingivitis_masks_dir)
        run([sys.executable, run_model,
             "--weights", args.gingivitis_weights,
             "--input", relevant_dir,
             "--output", gingivitis_masks_dir],
            "Stage 3 — Gingivitis segmentation")

        # Stage 4: overlay + compare predicted masks vs ground truth
        _create_final_results(args.images, gingivitis_masks_dir, args.masks, args.results)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
