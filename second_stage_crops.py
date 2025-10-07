# make_crops_all_nonblack.py
import cv2
import os
import sys
import numpy as np

def create_image_in_dir(aug_img, aug_mask, img_name, dest_root):
    out_img_dir = os.path.join(dest_root, "crops_images")
    out_msk_dir = os.path.join(dest_root, "mask_crops_images")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_img_dir, img_name), aug_img)
    cv2.imwrite(os.path.join(out_msk_dir, img_name), aug_mask)

def _grid_indices(L, win, stride, include_edge=True):
    """Start indices for a sliding window of size 'win' with step 'stride'."""
    xs = list(range(0, max(L - win + 1, 1), stride))
    if include_edge and xs and xs[-1] != L - win:
        xs.append(L - win)
    if not xs:  # image smaller than window
        xs = [0]
    return xs

def make_all_nonblack_crops(
    imgpath,
    maskpath,
    dest_root,
    xsize=256,
    ysize=256,
    stride_x=None,           # default: half overlap
    stride_y=None,
    min_nonblack_frac=0.75,  # keep iff ≥75% non-black
    black_thr=3,             # pixel is black if all channels ≤ thr
    include_edges=True       # force last crop flush with right/bottom edges
):
    image = cv2.imread(imgpath)                  # BGR
    mask  = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {imgpath}")
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {maskpath}")

    H, W = image.shape[:2]
    if W < xsize or H < ysize:
        print(f"[!] Skipping {os.path.basename(imgpath)} (image smaller than crop size)")
        return 0

    if stride_x is None: stride_x = max(1, xsize // 2)
    if stride_y is None: stride_y = max(1, ysize // 2)

    base = os.path.basename(imgpath)
    name, ext = os.path.splitext(base)

    # Boolean map: True where pixel is NOT near-black (any channel > thr)
    nonblack = np.any(image > black_thr, axis=2).astype(np.uint8)

    # Integral image for O(1) window sums (shape: (H+1, W+1))
    ii = cv2.integral(nonblack)  # int32/64

    total_pix = xsize * ysize
    min_nonblack_pix = int(np.ceil(min_nonblack_frac * total_pix))

    xs = _grid_indices(W, xsize, stride_x, include_edge=include_edges)
    ys = _grid_indices(H, ysize, stride_y, include_edge=include_edges)

    written = 0
    for y in ys:
        y1, y2 = y, y + ysize
        for x in xs:
            x1, x2 = x, x + xsize

            # Sum of non-black pixels in this window via integral image
            s = int(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])
            if s >= min_nonblack_pix:
                img_crop  = image[y1:y2, x1:x2]
                mask_crop = mask[y1:y2, x1:x2]
                create_image_in_dir(img_crop, mask_crop, f"{name}_nb_{written:05d}{ext}", dest_root)
                written += 1

    print(f"  -> {os.path.basename(imgpath)}: wrote {written} crops (≥{int(min_nonblack_frac*100)}% non-black)")
    return written

def main():
    if len(sys.argv) != 4:
        print("Usage: python make_crops_all_nonblack.py pics_dir masks_dir dest_dir")
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

    total = 0
    for i, filename in enumerate(files, 1):
        img_path  = os.path.join(pics_dir, filename)
        mask_path = os.path.join(masks_dir, filename)

        if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
            print(f"[{i}/{len(files)}] Skipping {filename} - missing image or mask")
            continue

        print(f"[{i}/{len(files)}] {filename}")
        try:
            total += make_all_nonblack_crops(
                img_path, mask_path, dest_root=dest_dir,
                xsize=256, ysize=256,
                stride_x=128, stride_y=128,       # 50% overlap; set to 1 for literal “all” positions
                min_nonblack_frac=0.75,
                black_thr=3,
                include_edges=True
            )
        except Exception as e:
            print(f"[✗] Error {filename}: {e}")

    print(f"Done. Wrote {total} crops.")

if __name__ == "__main__":
    main()
