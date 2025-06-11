import cv2
import os
import sys
import random

def create_image_in_dir(aug_img, aug_mask, img_name):
    output_dir_img = os.path.join(sys.argv[3], "crops_images")
    output_path_img = os.path.join(output_dir_img, img_name)
    cv2.imwrite(output_path_img, aug_img)

    output_dir_mask = os.path.join(sys.argv[3], "mask_crops_images")
    output_path_mask = os.path.join(output_dir_mask, img_name)
    cv2.imwrite(output_path_mask, aug_mask)


def make_crops(imgpath, maskpath, maxcorrelation, xsize, ysize, numofcrops):
    image = cv2.imread(imgpath)
    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)  # Force grayscale for binary mask

    if image is None:
        raise FileNotFoundError(f"Could not read image: {imgpath}")
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {maskpath}")

    height, width = image.shape[:2]

    xarr = [random.randint(0, width - xsize) for _ in range(numofcrops)]
    yarr = [random.randint(0, height - ysize) for _ in range(numofcrops)]

    kept = []
    for i in range(len(xarr)):
        xi, yi = xarr[i], yarr[i]
        overlap = False
        for xj, yj in kept:
            if abs(xi - xj) < maxcorrelation * xsize and abs(yi - yj) < maxcorrelation * ysize:
                overlap = True
                break
        if not overlap:
            kept.append((xi, yi))

    base_name = os.path.basename(imgpath)
    name, ext = os.path.splitext(base_name)

    for idx, (x, y) in enumerate(kept):
        crop = image[y:y+ysize, x:x+xsize]
        mask_crop = mask[y:y+ysize, x:x+xsize]

        white_pixels = cv2.countNonZero(mask_crop)
        total_pixels = xsize * ysize
        white_ratio = white_pixels / total_pixels

        if white_ratio < 0.10:
            continue  

        create_image_in_dir(crop, mask_crop, f"{name}_{idx}{ext}")



def main():
    if len(sys.argv) != 4:
        print("Usage: python make_crops.py pics_dir masks_dir dest_dir")
        sys.exit(1)

    for d in sys.argv[1:4]:
        if not os.path.isdir(d):
            print(f"Error: {d} is not a valid directory.")
            sys.exit(1)

    os.makedirs(os.path.join(sys.argv[3], "crops_images"), exist_ok=True)
    os.makedirs(os.path.join(sys.argv[3], "mask_crops_images"), exist_ok=True)

    for filename in os.listdir(sys.argv[1]):
        file_path = os.path.join(sys.argv[1], filename)
        file_mask_path = os.path.join(sys.argv[2], filename)
        make_crops(file_path, file_mask_path, 0.75, 512, 512, 40)
        print(f"[✓] Created crops for {filename}")


if __name__ == "__main__":
    main()
