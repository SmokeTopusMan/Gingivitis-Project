import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def run(cmd, stage_name):
    print(f"\n[{stage_name}] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {stage_name} failed (exit code {e.returncode})")
        sys.exit(1)


def create_final_results(images_dir, gingivitis_masks_dir, results_dir):
    """Green outline of the model's gingivitis prediction on each original image.

    Mirrors GingivitisApp._create_final_results in UI/main_screen.py.
    """
    os.makedirs(results_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(IMAGE_EXTS)]

    print(f"\nCreating final results with green outline for {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        mask_name = os.path.splitext(img_file)[0] + ".jpg"
        mask_path = os.path.join(gingivitis_masks_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {img_file}, copying original")
            img = Image.open(img_path)
            img.save(os.path.join(results_dir, img_file))
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

    print(f"Results saved to: {results_dir}")


def create_comparison(images_dir, gingivitis_masks_dir, dentist_masks_dir, comparison_dir):
    """Model-vs-dentist overlay (red=agree, orange=model only, green=dentist only)
    with a per-image stats box.

    Mirrors GingivitisApp._create_comparison in UI/main_screen.py.
    """
    os.makedirs(comparison_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(IMAGE_EXTS)]

    print(f"\nCreating comparison images for {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)

        # Model mask produced by the gingivitis model
        model_mask_path = os.path.join(gingivitis_masks_dir,
                                       os.path.splitext(img_file)[0] + ".jpg")

        # Dentist mask — try exact filename first, then other extensions
        dentist_mask_path = os.path.join(dentist_masks_dir, img_file)
        if not os.path.exists(dentist_mask_path):
            stem = os.path.splitext(img_file)[0]
            for ext in IMAGE_EXTS:
                candidate = os.path.join(dentist_masks_dir, stem + ext)
                if os.path.exists(candidate):
                    dentist_mask_path = candidate
                    break

        if not os.path.exists(model_mask_path):
            print(f"Warning: no model mask for {img_file}, skipping")
            continue
        if not os.path.exists(dentist_mask_path):
            print(f"Warning: no dentist mask for {img_file}, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        model_mask = Image.open(model_mask_path).convert("L")
        dentist_mask = Image.open(dentist_mask_path).convert("L")

        if model_mask.size != img.size:
            model_mask = model_mask.resize(img.size, Image.LANCZOS)
        if dentist_mask.size != img.size:
            dentist_mask = dentist_mask.resize(img.size, Image.LANCZOS)

        img_arr = np.array(img, dtype=np.float32)
        model_arr = np.array(model_mask)
        dentist_arr = np.array(dentist_mask)

        _, model_bin = cv2.threshold(model_arr, 127, 255, cv2.THRESH_BINARY)
        _, dentist_bin = cv2.threshold(dentist_arr, 127, 255, cv2.THRESH_BINARY)

        model_bool = model_bin > 0
        dentist_bool = dentist_bin > 0

        intersection = model_bool & dentist_bool     # red
        model_only   = model_bool & ~dentist_bool    # orange
        dentist_only = dentist_bool & ~model_bool    # green

        alpha = 0.45
        red    = np.array([255,   0,   0], dtype=np.float32)
        orange = np.array([255, 140,   0], dtype=np.float32)
        green  = np.array([  0, 210,   0], dtype=np.float32)

        result = img_arr.copy()
        result[intersection] = (1 - alpha) * result[intersection] + alpha * red
        result[model_only]   = (1 - alpha) * result[model_only]   + alpha * orange
        result[dentist_only] = (1 - alpha) * result[dentist_only] + alpha * green

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Compute percentages relative to total detected area
        total = int(intersection.sum() + model_only.sum() + dentist_only.sum())
        if total > 0:
            inter_pct   = intersection.sum() / total * 100
            model_pct   = model_only.sum()   / total * 100
            dentist_pct = dentist_only.sum() / total * 100
        else:
            inter_pct = model_pct = dentist_pct = 0.0

        # Draw stats box in top-right corner
        H_img, W_img = result.shape[:2]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, W_img / 2000)
        thickness  = max(1, int(font_scale * 2))
        lines = [
            (f"Intersection: {inter_pct:.1f}%",   (255,   0,   0)),
            (f"Model only:   {model_pct:.1f}%",   (255, 140,   0)),
            (f"Dentist only: {dentist_pct:.1f}%", (  0, 210,   0)),
        ]
        pad    = int(12 * font_scale)
        line_h = int(cv2.getTextSize("A", font, font_scale, thickness)[0][1] * 2.2)
        box_w  = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t, _ in lines) + pad * 2
        box_h  = line_h * len(lines) + pad
        margin = 20
        x0 = W_img - box_w - margin
        y0 = margin

        # Semi-transparent dark background
        overlay = result.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)

        # Text lines in matching colors (BGR for OpenCV)
        for i, (text, rgb) in enumerate(lines):
            bgr = (rgb[2], rgb[1], rgb[0])
            tx = x0 + pad
            ty = y0 + pad + line_h * i + line_h // 2
            cv2.putText(result, text, (tx, ty), font, font_scale, bgr, thickness, cv2.LINE_AA)

        Image.fromarray(result).save(os.path.join(comparison_dir, img_file))

    print(f"Comparison images saved to: {comparison_dir}")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Gingivitis detection pipeline — headless version of main_screen.py"
    )
    parser.add_argument("--images", required=True,
                        help="Directory containing input images")
    parser.add_argument("--masks", required=True,
                        help="Directory containing hand-made (dentist) gingivitis masks")
    parser.add_argument("--results", required=True,
                        help="Output directory (Results/ and Comparison/ are created inside)")
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

    results_dir = os.path.join(args.results, "Results")
    comparison_dir = os.path.join(args.results, "Comparison")
    os.makedirs(args.results, exist_ok=True)

    run_model = os.path.join(project_root, "tools", "run_model.py")
    get_relevant = os.path.join(project_root, "tools", "get_relevant.py")

    temp_dir = tempfile.mkdtemp(prefix="gingivitis_lambda_")
    teeth_masks_dir = os.path.join(temp_dir, "teeth_masks")
    relevant_dir = os.path.join(temp_dir, "relevant_images")
    gingivitis_masks_dir = os.path.join(temp_dir, "gingivitis_masks")

    try:
        # Stage 1: teeth segmentation  (mirrors _run_model with Teeth weights)
        os.makedirs(teeth_masks_dir)
        run([sys.executable, run_model,
             "--weights", args.teeth_weights,
             "--input", args.images,
             "--output", teeth_masks_dir],
            "Stage 1 — Teeth segmentation")

        # Stage 2: crop images to relevant dental region  (mirrors _run_get_relevant)
        # get_relevant.py writes to {argv[3]}/relevant_images/
        run([sys.executable, get_relevant,
             args.images, teeth_masks_dir, temp_dir],
            "Stage 2 — Relevant region extraction")

        # Stage 3: gingivitis segmentation on cropped images  (mirrors _run_model with Gingivitis weights)
        os.makedirs(gingivitis_masks_dir)
        run([sys.executable, run_model,
             "--weights", args.gingivitis_weights,
             "--input", relevant_dir,
             "--output", gingivitis_masks_dir],
            "Stage 3 — Gingivitis segmentation")

        # Stage 4: green-contour results  (mirrors _create_final_results)
        create_final_results(args.images, gingivitis_masks_dir, results_dir)

        # Stage 5: model-vs-dentist comparison  (mirrors _create_comparison)
        create_comparison(args.images, gingivitis_masks_dir, args.masks, comparison_dir)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone.")
    print(f"  Annotated images  ->  {results_dir}")
    print(f"  Model vs. dentist ->  {comparison_dir}")


if __name__ == "__main__":
    main()
