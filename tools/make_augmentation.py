import cv2
import albumentations as A
import matplotlib.pyplot as plt
import os
import sys


def create_image_in_dir(aug_img, aug_mask, img_name):
    output_dir = f"{sys.argv[3]}\\augmented_images"
    output_path_img = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path_img, aug_img)
    output_dir = f"{sys.argv[3]}\\mask_augmented_images"
    output_path_mask = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path_mask, aug_mask)


def make_augmented(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    transform = A.Compose([])
    mask_t = transform(image=mask)["image"]
    aug_img = transform(image=image)["image"]
    create_image_in_dir(aug_img, mask_t, f"{name}{ext}")

    transform = A.Compose([A.HorizontalFlip(p=1)])
    aug_img = transform(image=image)["image"]
    flipped_mask = transform(image=mask)["image"]
    create_image_in_dir(aug_img, flipped_mask, f"{name}_flipped{ext}")

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1)
    ])
    aug_img = transform(image=image)["image"]
    create_image_in_dir(aug_img, mask_t, f"{name}_Contrast_&_hue{ext}")

    transform = A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1)
    ])
    aug_img = transform(image=image)["image"]
    create_image_in_dir(aug_img, mask_t, f"{name}_CLAHE_&_RGBshift{ext}")

    # transform = A.Compose([
    #     A.RandomGamma(gamma_limit=(80, 120), p=1),
    # ])
    # aug_img = transform(image=image)["image"]
    # create_image_in_dir(aug_img, mask_t, f"{name}_gamma{ext}")



def main():
    if len(sys.argv) != 4:
        print("Usage: python make_augmentation.py pics_dir masks_dir dest_dir")
        sys.exit(1)

    if not os.path.isdir(sys.argv[1]):
        print(f"Error: {sys.argv[1]} is not a valid directory.")
        sys.exit(1)

    if not os.path.isdir(sys.argv[2]):
        print(f"Error: {sys.argv[2]} is not a valid directory.")
        sys.exit(1)

    if not os.path.isdir(sys.argv[3]):
        print(f"Error: {sys.argv[3]} is not a valid directory.")
        sys.exit(1)

    os.makedirs(f"{sys.argv[3]}\\augmented_images", exist_ok=True)
    os.makedirs(f"{sys.argv[3]}\\mask_augmented_images", exist_ok=True)

    for filename in os.listdir(sys.argv[1]):
        file_path = os.path.join(sys.argv[1], filename)
        file_mask_path = os.path.join(sys.argv[2], filename)
        make_augmented(file_path, file_mask_path)
        print(f"[✓] Created augmentation for {filename}")


if __name__ == "__main__":
    main()