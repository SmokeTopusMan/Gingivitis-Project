import os
import sys
import shutil
import subprocess
import numpy as np
from PIL import Image


def check_dependencies():
    """
    Check that required packages are installed.
    If something is missing, print a clear message and exit.
    """
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'segmentation_models_pytorch': 'segmentation-models-pytorch',
        'albumentations': 'albumentations',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'numpy': 'numpy'
    }

    missing_packages = []
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print("\n[ERROR] The following required packages are missing:\n")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print(
            "\nPlease install them using:\n"
            f"  pip install {' '.join(missing_packages)}\n"
        )
        sys.exit(1)


def get_directory_size(path: str) -> int:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def run_model(input_dir: str, weights: str) -> bool:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Treat Gingivitis-Project as project root
    project_root = script_dir

    if weights == "best_model.pth":
        temp_dir = os.path.join(script_dir, "temp_teeth_model")
        weights_path = os.path.join(project_root, "weights&results", "best_model.pth")
    else:
        temp_dir = os.path.join(script_dir, "temp_gingivitis_model")
        weights_path = os.path.join(project_root, "best_model_ging_clean.pth")

    os.makedirs(temp_dir, exist_ok=True)
    model_script = os.path.join(project_root, "tools", "run_model.py")

    cmd = [
        sys.executable,
        model_script,
        "--weights", weights_path,
        "--input", input_dir,
        "--output", temp_dir
    ]

    try:
        model_name = weights.replace("_weights.pth", "").replace("_model", "")
        print(f"\n[INFO] Running {model_name} model:")
        print("       " + " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error running model with weights {weights}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error while running model: {str(e)}")
        return False


def run_get_relevant(original_input_dir: str) -> bool:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp_teeth_model")

    # Treat Gingivitis-Project as project root
    project_root = script_dir
    get_relevant_script = os.path.join(project_root, "tools", "get_relevant.py")

    cmd = [
        sys.executable,
        get_relevant_script,
        original_input_dir,
        temp_dir,
        script_dir
    ]

    try:
        print("\n[INFO] Running get_relevant.py:")
        print("       " + " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        old_dir = os.path.join(script_dir, "relevant_images")
        new_dir = os.path.join(script_dir, "temp_relevant_images")

        if os.path.exists(old_dir):
            if os.path.exists(new_dir):
                shutil.rmtree(new_dir)
            os.rename(old_dir, new_dir)
            print("[INFO] Renamed 'relevant_images' to 'temp_relevant_images'")

        return True
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] Error running get_relevant.py:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in get_relevant: {str(e)}")
        return False
def run_postprocess_black_pixels(original_input_dir: str, masks_dir: str) -> str:
    """
    Run postprocessing: blacken out all mask pixels where the original image is black.
    Returns the directory containing postprocessed masks.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    postprocess_script = os.path.join(project_root, "tools", "post.py")
    out_dir = os.path.join(script_dir, "temp_gingivitis_model_post")

    cmd = [
        sys.executable,
        postprocess_script,
        "--images_dir", original_input_dir,
        "--masks_dir", masks_dir,
        "--out_dir", out_dir,
        "--thr", "3",
    ]

    try:
        print("\n[INFO] Running post.py:")
        print("       " + " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return out_dir
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] Error running postprocess_black_pixels.py:")
        print(e.stderr)
        # Fallback: just use original masks dir
        return masks_dir

def create_final_results(original_input_dir: str, masks_dir: str) -> bool:
    """
    Create green-overlay results and save them in a folder called 'preds'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gingivitis_masks_dir = masks_dir

    # Save to 'preds'
    final_results_dir = os.path.join(script_dir, "preds")
    os.makedirs(final_results_dir, exist_ok=True)

    try:
        image_files = [
            f for f in os.listdir(original_input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

        print(f"\n[INFO] Creating final results with green overlay for {len(image_files)} images...")
        print(f"[INFO] Output folder: {final_results_dir}")

        for img_file in image_files:
            img_path = os.path.join(original_input_dir, img_file)
            mask_name = os.path.splitext(img_file)[0] + ".png"
            mask_path = os.path.join(gingivitis_masks_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"[WARN] No mask found for {img_file}, copying original")
                img = Image.open(img_path)
                output_path = os.path.join(final_results_dir, img_file)
                img.save(output_path)
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if img.size != mask.size:
                mask = mask.resize(img.size, Image.LANCZOS)

            img_array = np.array(img)
            mask_array = np.array(mask)

            white_pixels = mask_array > 127

            overlay = img_array.copy()
            overlay[white_pixels] = [0, 255, 0]

            alpha = 0.4
            result = img_array.copy()
            result[white_pixels] = (
                alpha * overlay[white_pixels] +
                (1 - alpha) * img_array[white_pixels]
            ).astype(np.uint8)

            result_img = Image.fromarray(result)
            output_path = os.path.join(final_results_dir, img_file)
            result_img.save(output_path)

        print(f"\n[INFO] Final results saved to: {final_results_dir}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to create final results: {str(e)}")
        return False


def cleanup_temp_directories():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dirs = [
        os.path.join(script_dir, "temp_teeth_model"),
        os.path.join(script_dir, "temp_relevant_images"),
        os.path.join(script_dir, "temp_gingivitis_model"),
        os.path.join(script_dir, "temp_gingivitis_model_post")

    ]

    try:
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[INFO] Deleted temporary directory: {temp_dir}")

        print("[INFO] Cleanup completed successfully")
        return True

    except Exception as e:
        print(f"[WARN] Could not clean up some temporary directories: {str(e)}")
        return False


def process_directory(path: str):
    if not os.path.isdir(path):
        print(f"\n[ERROR] '{path}' is not a valid directory.")
        sys.exit(1)

    print(f"\n[INFO] Using input directory: {path}")

    try:
        dir_size = get_directory_size(path)
        required_space = dir_size * 3

        current_drive = os.path.abspath(__file__).split(os.sep)[0] + os.sep
        free_space = shutil.disk_usage(current_drive).free

        if free_space < required_space:
            required_gb = required_space / (1024 ** 3)
            available_gb = free_space / (1024 ** 3)
            print(
                "\n[ERROR] Not enough space for the model to work!"
                f"\n        Required:  {required_gb:.2f} GB"
                f"\n        Available: {available_gb:.2f} GB\n"
            )
            sys.exit(1)
        else:
            print("[INFO] Disk space check passed.")

        # 1) Run teeth model
        if not run_model(path, "best_model.pth"):
            print("\n[ERROR] Teeth model failed. Aborting.")
            sys.exit(1)

        # 2) Run get_relevant
        if not run_get_relevant(path):
            print("\n[ERROR] get_relevant failed. Aborting.")
            sys.exit(1)

        # 3) Run gingivitis model on temp_relevant_images
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_relevant_dir = os.path.join(script_dir, "temp_relevant_images")

        if not os.path.isdir(temp_relevant_dir):
            print(f"\n[ERROR] temp_relevant_images directory not found at {temp_relevant_dir}")
            sys.exit(1)

        if not run_model(temp_relevant_dir, "best_model_ging_clean.pth"):
            print("\n[ERROR] Gingivitis model failed. Aborting.")
            sys.exit(1)
        gingivitis_masks_dir = os.path.join(script_dir, "temp_gingivitis_model")
        post_masks_dir = run_postprocess_black_pixels(path, gingivitis_masks_dir)
        # 4) Create final overlays -> preds/
        
        if not create_final_results(path, masks_dir=post_masks_dir):
            print("\n[ERROR] Failed to create final results. Aborting.")
            sys.exit(1)

        # 5) Cleanup
        cleanup_temp_directories()

        print("\n[SUCCESS] Processing completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {str(e)}")
        sys.exit(1)


def main():
    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] argv:", sys.argv)

    check_dependencies()

    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    else:
        input_dir = input("Enter the path of the directory with the images: ").strip()

    if not input_dir:
        print("[ERROR] No directory provided.")
        sys.exit(1)

    process_directory(input_dir)


if __name__ == "__main__":
    main()
