import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image


def _create_final_results(images_dir, gingivitis_masks_dir, results_dir):
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    print(f"\n[Stage 4] Overlaying results for {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        mask_name = os.path.splitext(img_file)[0] + ".jpg"
        mask_path = os.path.join(gingivitis_masks_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"  [!] No gingivitis mask for {img_file}, copying original")
            shutil.copy2(img_path, os.path.join(results_dir, img_file))
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.LANCZOS)

        img_array = np.array(img)
        mask_array = np.array(mask)

        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = img_array.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), thickness=3)

        Image.fromarray(result).save(os.path.join(results_dir, img_file))
        print(f"  [OK] {img_file}")

    print(f"Final results saved to: {results_dir}")


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
                        help="Directory where Stage 1 teeth masks will be saved")
    parser.add_argument("--results", required=True,
                        help="Directory where final annotated images will be saved")
    parser.add_argument("--teeth-weights",
                        default=os.path.join(project_root, "weights&results", "Teeth_model_weights.pth"),
                        help="Path to teeth segmentation model weights")
    parser.add_argument("--gingivitis-weights",
                        default=os.path.join(project_root, "weights&results", "Gingivitis_model_weights.pth"),
                        help="Path to gingivitis segmentation model weights")
    args = parser.parse_args()

    # Validate inputs
    for path, name in [(args.images, "--images"),
                       (args.teeth_weights, "--teeth-weights"),
                       (args.gingivitis_weights, "--gingivitis-weights")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} path does not exist: {path}")
            sys.exit(1)

    os.makedirs(args.masks, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    run_model = os.path.join(project_root, "tools", "run_model.py")
    get_relevant = os.path.join(project_root, "tools", "get_relevant.py")

    temp_dir = tempfile.mkdtemp(prefix="gingivitis_lambda_")
    relevant_dir = os.path.join(temp_dir, "relevant_images")
    gingivitis_masks_dir = os.path.join(temp_dir, "gingivitis_masks")

    try:
        # Stage 1: teeth segmentation
        run([sys.executable, run_model,
             "--weights", args.teeth_weights,
             "--input", args.images,
             "--output", args.masks],
            "Stage 1 — Teeth segmentation")

        # Stage 2: crop to relevant region
        # get_relevant.py writes to {sys.argv[3]}/relevant_images/
        run([sys.executable, get_relevant,
             args.images, args.masks, temp_dir],
            "Stage 2 — Relevant region extraction")

        # Stage 3: gingivitis segmentation on cropped images
        os.makedirs(gingivitis_masks_dir, exist_ok=True)
        run([sys.executable, run_model,
             "--weights", args.gingivitis_weights,
             "--input", relevant_dir,
             "--output", gingivitis_masks_dir],
            "Stage 3 — Gingivitis segmentation")

        # Stage 4: overlay green contours on originals
        _create_final_results(args.images, gingivitis_masks_dir, args.results)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
