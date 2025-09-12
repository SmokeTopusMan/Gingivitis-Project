# make_crops_combo.py
import cv2
import os
import sys
import random

def create_image_in_dir(aug_img, aug_mask, img_name, dest_root):
    out_img_dir = os.path.join(dest_root, "crops_images")
    out_msk_dir = os.path.join(dest_root, "mask_crops_images")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_img_dir, img_name), aug_img)
    cv2.imwrite(os.path.join(out_msk_dir, img_name), aug_mask)

def make_crops_combo(
    imgpath,
    maskpath,
    dest_root,
    maxcorrelation=0.75,
    xsize=512,
    ysize=512,
    target_pos=10,            # crops with objects in mask
    target_pureblack=5,       # crops with (near) pure-black mask
    pos_min_white=0.15,       # 15%..90% white pixels in mask
    pos_max_white=0.90,
    pureblack_max_white=0.05, # ≤5% white pixels in mask
    max_attempts=100000
):
    """
    Random crops with overlap control. Keeps only crops where the IMAGE crop
    is not entirely black. Fills two buckets per image:
      - positives (mask has objects: white ratio in [pos_min_white, pos_max_white])
      - pureblack negatives (mask nearly empty: white ratio ≤ pureblack_max_white)
    """
    image = cv2.imread(imgpath)
    mask  = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {imgpath}")
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {maskpath}")

    h, w = image.shape[:2]
    if w < xsize or h < ysize:
        print(f"[!] Skipping {os.path.basename(imgpath)} (image smaller than crop size)")
        return

    base = os.path.basename(imgpath)
    name, ext = os.path.splitext(base)

    # Track accepted crop top-lefts to enforce overlap across BOTH buckets
    accepted_coords = []
    pos_coords = []
    neg_coords = []

    attempts = 0
    needed = lambda: (len(pos_coords) < target_pos) or (len(neg_coords) < target_pureblack)

    while needed() and attempts < max_attempts:
        x = random.randint(0, w - xsize)
        y = random.randint(0, h - ysize)

        # Overlap control (same as your style)
        overlap = False
        for xp, yp in accepted_coords:
            if abs(x - xp) < maxcorrelation * xsize and abs(y - yp) < maxcorrelation * ysize:
                overlap = True
                break
        if overlap:
            attempts += 1
            continue

        # Crop image and mask
        img_crop  = image[y:y+ysize, x:x+xsize]
        mask_crop = mask[y:y+ysize, x:x+xsize]

        # 1) NEW RULE: image crop must NOT be all black
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(gray) == 0:
            attempts += 1
            continue

        # 2) Classify by mask white ratio
        white_pixels = cv2.countNonZero(mask_crop)
        total_pixels = xsize * ysize
        white_ratio = white_pixels / total_pixels

        # Try to fill positives first if still needed and condition matches
        placed = False
        if len(pos_coords) < target_pos and (pos_min_white <= white_ratio <= pos_max_white):
            pos_coords.append((x, y))
            accepted_coords.append((x, y))
            placed = True

        # Otherwise try to fill pure-black bucket if condition matches
        if (not placed) and len(neg_coords) < target_pureblack and (white_ratio <= pureblack_max_white):
            neg_coords.append((x, y))
            accepted_coords.append((x, y))
            placed = True

        # If it matches neither desired bucket (or the bucket is full), discard and continue
        attempts += 1

    # Write crops (keep names compact and collision-free)
    idx = 0
    for (x, y) in pos_coords:
        img_crop  = image[y:y+ysize, x:x+xsize]
        mask_crop = mask[y:y+ysize, x:x+xsize]
        create_image_in_dir(img_crop, mask_crop, f"{name}_pos_{idx}{ext}", dest_root)
        idx += 1

    jdx = 0
    for (x, y) in neg_coords:
        img_crop  = image[y:y+ysize, x:x+xsize]
        mask_crop = mask[y:y+ysize, x:x+xsize]
        create_image_in_dir(img_crop, mask_crop, f"{name}_neg_{jdx}{ext}", dest_root)
        jdx += 1

    print(f"  -> {os.path.basename(imgpath)}: wrote {len(pos_coords)} pos, {len(neg_coords)} neg")

def main():
    if len(sys.argv) != 4:
        print("Usage: python make_crops_combo.py pics_dir masks_dir dest_dir")
        sys.exit(1)

    pics_dir, masks_dir, dest_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    for d in (pics_dir, masks_dir, dest_dir):
        if not os.path.isdir(d):
            print(f"Error: {d} is not a valid directory.")
            sys.exit(1)

    os.makedirs(os.path.join(dest_dir, "crops_images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "mask_crops_images"), exist_ok=True)

    files = [f for f in os.listdir(pics_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    print(f"Processing {len(files)} images...")

    for i, filename in enumerate(files, 1):
        img_path  = os.path.join(pics_dir, filename)
        mask_path = os.path.join(masks_dir, filename)

        if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
            print(f"[{i}/{len(files)}] Skipping {filename} - missing image or mask")
            continue

        print(f"[{i}/{len(files)}] {filename}")
        try:
            make_crops_combo(
                img_path,
                mask_path,
                dest_root=dest_dir,
                maxcorrelation=0.75,
                xsize=512,
                ysize=512,
                target_pos=10,          # <- adjust per your dataset
                target_pureblack=5,     # <- adjust per your dataset
                pos_min_white=0.15,
                pos_max_white=0.90,
                pureblack_max_white=0.05,
                max_attempts=100000
            )
        except Exception as e:
            print(f"[✗] Error {filename}: {e}")

if __name__ == "__main__":
    main()
