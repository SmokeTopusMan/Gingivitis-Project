import cv2
import os
import sys
import numpy as np
import gc
from __future__ import annotations
import numpy as np
from typing import Iterable, Optional, Tuple

from typing import Optional, Tuple, Iterable
import numpy as np

from typing import Optional, Tuple, Iterable
import numpy as np
def _fill_nan_nearest(arr, fill_val):
    x = np.arange(arr.size)
    m = ~np.isnan(arr)
    if m.sum() == 0:
        return np.full_like(arr, fill_val, dtype=float)
    if m.sum() == 1:
        return np.full_like(arr, arr[m][0], dtype=float)
    return np.interp(x, x[m], arr[m])

def _moving_average(a, k=15):
    if k <= 1: return a
    k = int(max(1, k))
    pad = k//2
    a_pad = np.pad(a, (pad, pad), mode='edge')
    kernel = np.ones(k, dtype=float)/k
    return np.convolve(a_pad, kernel, mode='valid')

def row_side_mask_from_straightness(white_mask):
    """
    white_mask: bool array (H,W). Returns:
      side  : 'top' or 'bottom' (less-straight boundary)
      side_mask: bool (H,W) True on the chosen side of the component
    Assumes the row is roughly horizontal. (For large rotations, rotate by PCA first.)
    """
    H, W = white_mask.shape
    has_col = white_mask.any(axis=0)

    # top_y[x] = smallest y with white at column x; bottom_y[x] = largest y
    top_y = np.full(W, np.nan, dtype=float)
    bottom_y = np.full(W, np.nan, dtype=float)
    # top: first True index
    first_idx = np.argmax(white_mask, axis=0)  # 0 if none; guard with has_col
    top_y[has_col] = first_idx[has_col]
    # bottom: last True index
    rev_first = np.argmax(np.flipud(white_mask), axis=0)
    bottom_y[has_col] = (H - 1) - rev_first[has_col]

    # fill gaps and smooth a bit
    top_y    = _fill_nan_nearest(top_y, H/2)
    bottom_y = _fill_nan_nearest(bottom_y, H/2)
    top_y    = _moving_average(top_y, k=15)
    bottom_y = _moving_average(bottom_y, k=15)

    # straightness = residual std after best straight-line fit
    xs = np.arange(W, dtype=float)
    At = np.vstack([xs, np.ones_like(xs)]).T
    ab_t, *_ = np.linalg.lstsq(At, top_y, rcond=None)
    ab_b, *_ = np.linalg.lstsq(At, bottom_y, rcond=None)
    std_top = np.std(top_y    - (ab_t[0]*xs + ab_t[1]))
    std_bot = np.std(bottom_y - (ab_b[0]*xs + ab_b[1]))

    side = 'top' if std_top > std_bot else 'bottom'

    Y = np.arange(H)[:, None]  # (H,1)
    if side == 'top':
        side_mask = (Y < top_y[None, :])
    else:
        side_mask = (Y > bottom_y[None, :])

    return side, side_mask.astype(bool)
def hull_hollow_masks(white_mask: np.ndarray):
    """
    white_mask: bool or 0/1, shape (H,W). True/1 = white (teeth).
    Returns:
      inner_centroid_hollow: only the hollow component that contains the white centroid (bool)
      all_hull_gaps: all black pixels inside the convex hull but outside the white (bool)
    """
    wm = white_mask.astype(bool)
    white_u8 = wm.astype(np.uint8)

    # contours -> convex hull
    cnts, _ = cv2.findContours(white_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = white_u8.shape
        return np.zeros((h, w), bool), np.zeros((h, w), bool)

    pts = np.vstack(cnts)
    hull = cv2.convexHull(pts)
    hull_mask = np.zeros_like(white_u8)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    # all gaps inside hull (hull \ white)
    all_hull_gaps = hull_mask.astype(bool) & (~wm)

    # centroid of white
    ys, xs = np.where(wm)
    if ys.size == 0:
        return np.zeros_like(wm), all_hull_gaps

    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))
    cy = np.clip(cy, 0, wm.shape[0]-1)
    cx = np.clip(cx, 0, wm.shape[1]-1)

    # the single hollow connected to centroid (if centroid lies in a hull gap)
    inner_centroid_hollow = np.zeros_like(wm)
    if all_hull_gaps[cy, cx]:
        nlab, lab = cv2.connectedComponents(all_hull_gaps.astype(np.uint8), connectivity=8)
        inner_centroid_hollow = (lab == lab[cy, cx])

    return inner_centroid_hollow.astype(bool), all_hull_gaps.astype(bool)

class WhiteSet:
    """
    A set of white pixels stored as (y, x) coordinates.
    _pixels: np.ndarray of shape (N, 2), dtype=int32
    """

    def __init__(
        self,
        pixels: Optional[np.ndarray] = None,
        pixel1: Optional[Tuple[int, int]] = None
    ):
        if pixels is None:
            pixels = np.empty((0, 2), dtype=np.int32)

        if pixel1 is not None:
            p = np.array([[int(pixel1[0]), int(pixel1[1])]], dtype=np.int32)
            pixels = np.vstack([pixels.astype(np.int32, copy=False), p])

        pixels = pixels.astype(np.int32, copy=False)
        if pixels.size > 0:
            pixels = np.unique(pixels, axis=0)

        self._pixels = pixels
        self._recompute_bounds()

    # ----- getters -----
    def getmax(self) -> Optional[int]:
        return self._maxY

    def getmin(self) -> Optional[int]:
        return self._minY

    def getpixels(self) -> np.ndarray:
        return self._pixels

    # ----- helpers -----
    def is_empty(self) -> bool:
        return self._pixels.size == 0

    def _recompute_bounds(self) -> None:
        if self._pixels.size == 0:
            self._minY, self._maxY = None, None
        else:
            self._minY = int(self._pixels[:, 0].min())
            self._maxY = int(self._pixels[:, 0].max())

    def kill(self) -> None:
        self._pixels = np.empty((0, 2), dtype=np.int32)
        self._recompute_bounds()

    def add_pixels(self, coords: Iterable[Tuple[int, int]]) -> None:
        """Add many (y, x) pixels to the set."""
        coords = list(coords)
        if not coords:
            return
        to_add = np.array(coords, dtype=np.int32).reshape(-1, 2)
        if self._pixels.size == 0:
            self._pixels = np.unique(to_add, axis=0)
        else:
            self._pixels = np.unique(np.vstack([self._pixels, to_add]), axis=0)
        self._recompute_bounds()

    def Unite(self, setTwo: "WhiteSet", connectivity: int = 8, same_row_merge: bool = True) -> bool:
        """
        Merge setTwo into self if they touch (4/8-connectivity) OR (optionally) share any y-row.
        Returns True if a merge happened.
        """
        if setTwo.is_empty():
            return False
        if self.is_empty():
            # adopt setTwo
            self._pixels = setTwo._pixels.copy()
            self._recompute_bounds()
            setTwo.kill()
            return True

        # Optional fast-path: merge if any row (y) is shared
        if same_row_merge:
            if len(set(self._pixels[:, 0]).intersection(set(setTwo._pixels[:, 0]))) > 0:
                merged = np.vstack([self._pixels, setTwo._pixels])
                self._pixels = np.unique(merged, axis=0)
                self._recompute_bounds()
                setTwo.kill()
                return True

        # Adjacency test
        other_set = set(map(tuple, setTwo._pixels.tolist()))
        if connectivity == 4:
            nbrs = [(1,0), (-1,0), (0,1), (0,-1)]
        elif connectivity == 8:
            nbrs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        else:
            raise ValueError("connectivity must be 4 or 8")

        touches = False
        for (y, x) in self._pixels:
            if (y, x) in other_set:
                touches = True
                break
            for dy, dx in nbrs:
                if (y + dy, x + dx) in other_set:
                    touches = True
                    break
            if touches:
                break

        if not touches:
            return False

        merged = np.vstack([self._pixels, setTwo._pixels])
        self._pixels = np.unique(merged, axis=0)
        self._recompute_bounds()
        setTwo.kill()
        return True

def create_image_in_dir(aug_img, img_name):
    output_dir = os.path.join(sys.argv[3], "relevant_images")
    output_path_img = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path_img, aug_img)

def cut_to_the_chase_efficient(image_path, mask_path):
    """
    Keep pixels within 'cutcount' pixels of any white (255) pixel in the mask;
    black out white pixels and non-white pixels farther than that.
    Uses a distance transform instead of a huge dilation kernel,
    preserving the same effective radius as the original code.
    """
    image = cv2.imread(image_path)  # BGR
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error: Could not load {image_path} or {mask_path}")
        return

    h, w = image.shape[:2]
    # SAME radius as before (no cap)
    cutcount = int((w + h) * 0.05)
    if cutcount < 1:
        cutcount = 1

    base_name = os.path.basename(image_path)

    white_mask = (mask == 255)
    ys, xs = np.where(white_mask)

    sets = [WhiteSet(pixel1=(int(y), int(x))) for y, x in zip(ys, xs)]

    for j in range(len(sets)):
        if sets[j].is_empty():
            continue
        for k in range(j + 1, len(sets)):  # k>j avoids double work
            if sets[k].is_empty():
                continue
            sets[j].Unite(sets[k], connectivity=8, same_row_merge=True)

# Collect non-empty connected components
    components = [s for s in sets if not s.is_empty()]
    print(f"Found {len(components)} components")
        
    if not np.any(white_mask):
        print(f"Warning: No white pixels found in {mask_path}")
            
            
    # Distance from each NON-WHITE pixel to the nearest WHITE pixel (Euclidean)
    # Foreground = non-white (1), background = white (0)
    non_white_u8 = (mask != 255).astype(np.uint8)
    dist = cv2.distanceTransform(non_white_u8, cv2.DIST_L2, 3)

    # Pixels farther than the radius are "far from white"
    non_white_mask = (mask != 255)
    far_from_white_mask = (dist > cutcount) & non_white_mask
    pixels_to_black_out = white_mask | far_from_white_mask

    if len(components) == 2:
        smallest_maxy = h
        largest_miny = 0

        for c in components:
            if smallest_maxy > c.getmax():
                smallest_maxy = c.getmax()
            if largest_miny < c.getmin():
                largest_miny = c.getmin()

        middle_mask = np.zeros_like(non_white_u8, dtype=bool)
        if largest_miny >= smallest_maxy:  # ensure valid range
            middle_mask[smallest_maxy:largest_miny+1, :] = (non_white_u8[smallest_maxy:largest_miny+1, :] > 0)
    # Combine with the other masks
        pixels_to_black_out = white_mask | far_from_white_mask | middle_mask
    ys, xs = np.where(mask == 255)
    cy = int(np.mean(ys))
    cx = int(np.mean(xs))
    if len(components) ==1 and (mask[cy,cx] != 255) and (components[0].getmax()-components[0].getmin()>=0.7*h):
            inner_centroid_hollow, all_hull_gaps = hull_hollow_masks(white_mask)
            inner_centroid_hollow &= (dist <= cutcount)
            all_hull_gaps        &= (dist <= cutcount)
            inner_hollow=all_hull_gaps
            inner_hollow &= (dist <= cutcount)
            inner_hollow = inner_centroid_hollow | all_hull_gaps

            pixels_to_black_out = ~inner_hollow
    if len(components) ==1 and (components[0].getmax()-components[0].getmin()<0.4*h):
        side, side_mask = row_side_mask_from_straightness(white_mask)

        # Keep only near-white background on the chosen side.
        near_bg = (dist <= cutcount) & (~white_mask)
        chosen_bg = near_bg & side_mask

        # Black out everything else (white, far, and the "other" side)
        pixels_to_black_out = ~(chosen_bg)


    keep_count = np.sum(~pixels_to_black_out)
    print(f"Keeping {keep_count} pixels; cutcount={cutcount} (h={h}, w={w})")

    result = image.copy()
    result[pixels_to_black_out] = [0, 0, 0]

    # Save as PNG (forced by create_image_in_dir)
    create_image_in_dir(result, base_name)

    del image, mask, result, white_mask, non_white_u8, dist, non_white_mask, far_from_white_mask
    gc.collect()


def cut_to_the_chase_chunked(image_path, mask_path):
    """Alternative chunked version for exact distance calculations"""
    # Load images
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error: Could not load {image_path} or {mask_path}")
        return

    height, width = image.shape[:2]
    cutcount = ((width + height)) * 0.01

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # Get white pixel coordinates
    white_coords = np.column_stack(np.where(mask == 255))  # (y, x) format
    white_coords = white_coords[:, [1, 0]]  # Convert to (x, y) format

    # Get non-white pixel coordinates
    non_white_coords = np.column_stack(np.where(mask != 255))  # (y, x) format
    non_white_coords = non_white_coords[:, [1, 0]]  # Convert to (x, y) format

    if len(white_coords) == 0 or len(non_white_coords) == 0:
        print(f"Warning: No white or non-white pixels found in {mask_path}")
        return

    print(f"Processing {len(non_white_coords)} non-white pixels against {len(white_coords)} white pixels")

    # Process in chunks to avoid memory issues
    from scipy.spatial.distance import cdist
    chunk_size = 10000
    far_coords_list = []

    for i in range(0, len(non_white_coords), chunk_size):
        chunk = non_white_coords[i:i + chunk_size]

        # Calculate distances for this chunk
        distances = cdist(chunk, white_coords, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        # Find pixels that are FAR from white pixels (>= cutcount distance)
        far_mask = min_distances >= cutcount

        far_coords_chunk = chunk[far_mask]
        if len(far_coords_chunk) > 0:
            far_coords_list.append(far_coords_chunk)

        # Clean up memory
        del distances, min_distances, far_mask, far_coords_chunk
        gc.collect()

    # Combine all far coordinates
    if far_coords_list:
        far_coords = np.vstack(far_coords_list)
    else:
        far_coords = np.array([])

    print(f"Blacking out {len(white_coords)} white pixels")
    print(f"Found {len(far_coords)} non-white pixels to black out (far from white)")

    # Apply blackout: black out white pixels AND pixels far from white
    result = image.copy()

    # First, black out all white pixels
    result[mask == 255] = [0, 0, 0]

    # Then, black out non-white pixels that are far from white
    if len(far_coords) > 0:
        for x, y in far_coords:
            result[y, x] = [0, 0, 0]

    # Save the result
    output_filename = f"{name}{ext}"
    create_image_in_dir(result, output_filename)

    # Clean up memory
    del image, mask, result, white_coords, non_white_coords, far_coords
    gc.collect()

def main():
    if len(sys.argv) != 4:
        print("Usage: python getrelevant.py pics_dir masks_dir dest_dir")
        sys.exit(1)

    # Validate directories
    for i, arg_name in enumerate(["pics_dir", "masks_dir", "dest_dir"], 1):
        if not os.path.isdir(sys.argv[i]):
            print(f"Error: {sys.argv[i]} is not a valid directory ({arg_name}).")
            sys.exit(1)

    os.makedirs(os.path.join(sys.argv[3], "relevant_images"), exist_ok=True)

    # Get image files
    image_files = [f for f in os.listdir(sys.argv[1])
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    print(f"Processing {len(image_files)} images...")

    for i, filename in enumerate(image_files, 1):
        file_path = os.path.join(sys.argv[1], filename)
        file_mask_path = os.path.join(sys.argv[2], os.path.splitext(filename)[0] + ".png")

        print(f"[{i}/{len(image_files)}] Processing {filename}...")

        if os.path.isfile(file_path) and os.path.isfile(file_mask_path):
            try:
                # Use the efficient morphological version
                # Switch to cut_to_the_chase_chunked() if you need exact distance calculations
                cut_to_the_chase_efficient(file_path, file_mask_path)
                print(f"[✓] Completed {filename}")
            except Exception as e:
                print(f"[✗] Error processing {filename}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[!] Skipping {filename} - missing image or mask file")

        gc.collect()

if __name__ == "__main__":

    main()
