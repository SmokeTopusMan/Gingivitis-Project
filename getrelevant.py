from __future__ import annotations
import cv2
import os
import sys
import numpy as np
import gc
import numpy as np
from typing import Iterable, Optional, Tuple

from typing import Optional, Tuple, Iterable
import numpy as np

from typing import Optional, Tuple, Iterable
import numpy as np



def row_side_flatness_masks(white_mask, smooth_k=15):
    """
    Returns:
      top_mask, bottom_mask  (bool HxW)
      metrics = dict(std_top, std_bot, gap_abs, gap_rel)
    """
    H, W = white_mask.shape
    has_col = white_mask.any(axis=0)

    # top & bottom y per column
    top_y = np.full(W, np.nan, dtype=float)
    bot_y = np.full(W, np.nan, dtype=float)
    first_idx = np.argmax(white_mask, axis=0)
    top_y[has_col] = first_idx[has_col]
    rev_first = np.argmax(np.flipud(white_mask), axis=0)
    bot_y[has_col] = (H - 1) - rev_first[has_col]

    # fill gaps + smooth
    def _fill_nan_nearest(a, fill):
        x = np.arange(a.size)
        m = ~np.isnan(a)
        if m.sum() == 0:  return np.full_like(a, fill, dtype=float)
        if m.sum() == 1:  return np.full_like(a, a[m][0], dtype=float)
        return np.interp(x, x[m], a[m])

    def _movavg(a, k):
        if k <= 1: return a
        k = int(max(1, k)); pad = k//2
        ap = np.pad(a, (pad, pad), mode='edge')
        return np.convolve(ap, np.ones(k, dtype=float)/k, mode='valid')

    top_y = _movavg(_fill_nan_nearest(top_y, H/2), k=smooth_k)
    bot_y = _movavg(_fill_nan_nearest(bot_y, H/2), k=smooth_k)

    # straightness via linear fit residuals
    xs = np.arange(W, dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    a_t, b_t = np.linalg.lstsq(A, top_y, rcond=None)[0]
    a_b, b_b = np.linalg.lstsq(A, bot_y, rcond=None)[0]
    std_top = float(np.std(top_y - (a_t*xs + b_t)))
    std_bot = float(np.std(bot_y - (a_b*xs + b_b)))

    Y = np.arange(H)[:, None]
    top_mask    = (Y <  top_y[None, :])
    bottom_mask = (Y >  bot_y[None, :])

    gap_abs = abs(std_top - std_bot)
    denom   = max(std_top, std_bot, 1e-6)
    gap_rel = gap_abs / denom

    metrics = dict(std_top=std_top, std_bot=std_bot, gap_abs=gap_abs, gap_rel=gap_rel)
    return top_mask.astype(bool), bottom_mask.astype(bool), metrics

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
def binarize_mask(mask_gray: np.ndarray, use_otsu: bool = True, thr: int = 128) -> np.ndarray:
    """
    Return boolean mask where True means 'white object'.
    Uses Otsu by default; falls back to fixed threshold.
    """
    if use_otsu:
        # Otsu expects 8-bit; mask already is, but guard anyway
        _, m = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        white_mask = (m > 0)
    else:
        white_mask = (mask_gray >= thr)
    return white_mask
def ensure_object_is_white(white_mask: np.ndarray) -> np.ndarray:
    # Auto invert if most of the image is "white"
    return ~white_mask if white_mask.mean() > 0.6 else white_mask

class _YSpanComp:
    def __init__(self, miny, maxy):
        self._minY = int(miny); self._maxY = int(maxy)
    def getmin(self): return self._minY
    def getmax(self): return self._maxY
class CompSpan:
    def __init__(self, miny, maxy, minx, maxx):
        self._minY, self._maxY = int(miny), int(maxy)
        self._minX, self._maxX = int(minx), int(maxx)
    def getmin(self):  return self._minY
    def getmax(self):  return self._maxY
    def getminx(self): return self._minX
    def getmaxx(self): return self._maxX
    def yspan(self):   return self._maxY - self._minY
    def xspan(self):   return self._maxX - self._minX

def components_from_mask(
    white_mask: np.ndarray,
    connectivity: int = 8,
    y_connect: bool = True,     # <-- enable y-level connectivity
    y_gap: int = 0,             # rows of tolerance when merging (e.g. 0–3)
    bridge_px: int = 0,         # optional closing to bridge tiny cracks first
    min_area_frac: float = 0.01 # <-- NEW: drop comps smaller than 1% of image
):
    """
    Returns components with X/Y spans. If y_connect=True, components whose
    [miny,maxy] intervals overlap (within y_gap) are merged into one.
    Components with pixel area < (min_area_frac * H * W) are ignored.
    """
    m = white_mask.astype(np.uint8)

    # Optional: bridge tiny cracks before labeling (harmless if 0)
    if bridge_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bridge_px+1, 2*bridge_px+1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity)

    H, W = m.shape
    min_area = max(1, int(min_area_frac * H * W))

    comps = []
    for lab in range(1, num):  # 0 is background
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue  # skip tiny component

        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        comps.append([y, y+h-1, x, x+w-1])  # [miny, maxy, minx, maxx]

    if not y_connect or len(comps) <= 1:
        return [CompSpan(a, b, c, d) for (a, b, c, d) in comps]

    # ---- Y-level merge (interval union with tolerance y_gap) ----
    comps.sort(key=lambda t: t[0])            # sort by miny
    merged = []
    cur = comps[0]
    for ny0, ny1, nx0, nx1 in comps[1:]:
        if ny0 <= cur[1] + y_gap:
            cur[1] = max(cur[1], ny1)
            cur[2] = min(cur[2], nx0)
            cur[3] = max(cur[3], nx1)
        else:
            merged.append(cur)
            cur = [ny0, ny1, nx0, nx1]
    merged.append(cur)

    return [CompSpan(a, b, c, d) for (a, b, c, d) in merged]


def near_background_via_dt(white_mask: np.ndarray, cutcount: int) -> np.ndarray:
    """
    Returns boolean mask of non-white pixels within 'cutcount' pixels of white.
    Fast: computes DT only in an ROI around the whites.
    """
    H, W = white_mask.shape
    ys, xs = np.where(white_mask)
    if ys.size == 0:
        return np.zeros((H, W), dtype=bool)

    pad = int(cutcount) + 2
    y0 = max(0, ys.min() - pad); y1 = min(H, ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad); x1 = min(W, xs.max() + pad + 1)
    roi = (slice(y0, y1), slice(x0, x1))

    # DT expects 1=foreground, 0=background; we want distance from non-white to white
    inv_roi = (~white_mask[roi]).astype(np.uint8)  # 1 where non-white
    # Distance to nearest 0 (white) in the ROI
    dist_roi = cv2.distanceTransform(inv_roi, cv2.DIST_L2, 3)

    near = np.zeros((H, W), dtype=bool)
    # within radius AND actually non-white
    near[roi] = (dist_roi <= cutcount) & (~white_mask[roi])
    return near
def split_mask_if_row_bridge(white_mask: np.ndarray,
                             width_frac: float = 0.10,
                             both_frac: float = 0.50) -> tuple[np.ndarray, Optional[int]]:
    """
    If there's a row y that meets:
      - >= width_frac of columns are non-white
      - among those columns, >= both_frac have white somewhere above and somewhere below
    then return a copy of white_mask with that row zeroed (split line) and the y index.
    Otherwise, return (white_mask, None).
    """
    m = white_mask.astype(bool)
    H, W = m.shape
    if H < 3 or W == 0:
        return m, None

    # Cumulative sums per column for O(1) "any above/below" checks
    cum = np.cumsum(m.view(np.uint8), axis=0)          # HxW
    tot = cum[-1, :]                                   # 1xW (total white count per column)

    thresh_nonwhite = max(1, int(width_frac * W))

    best_y = None
    best_score = -1

    for y in range(1, H-1):  # avoid very top/bottom
        nonwhite = ~m[y, :]
        n_nw = int(nonwhite.sum())
        if n_nw < thresh_nonwhite:
            continue

        above_any = cum[y-1, :] > 0
        below_any = (tot - cum[y, :]) > 0

        both = nonwhite & above_any & below_any
        score = int(both.sum())

        # require >= 50% of the row's non-white columns to have white above & below
        if score >= int(both_frac * n_nw) and score > best_score:
            best_score = score
            best_y = y

    if best_y is None:
        return m, None

    m2 = m.copy()
    m2[best_y, :] = False   # split line
    return m2, best_y
def middle_band_by_row_test(white_mask: np.ndarray,
                            min_width_frac: float = 0.07,   # ≥7% of width non-white
                            min_between_frac: float = 0.45, # ≥45% of those columns have white above & below
                            min_band_rows: int = 5,         # need a contiguous band ≥ 5 rows
                            x0: int = None, x1: int = None):
    """
    Returns (band_mask: bool HxW, (top, bot) or None).
    band_mask True on the best 'between' band.
    """
    H, W = white_mask.shape

    # columns that actually contain any white (teeth)
    cols = np.where(white_mask.any(axis=0))[0]
    if cols.size == 0:
        return np.zeros_like(white_mask, bool), None
    if x0 is None or x1 is None:
        x0, x1 = int(cols[0]), int(cols[-1])
    x0 = max(0, min(x0, W-1)); x1 = max(0, min(x1, W-1))
    if x1 <= x0:
        return np.zeros_like(white_mask, bool), None

    width = (x1 - x0 + 1)

    # fast “white above / below” via cumulative sums
    above_cum = np.cumsum(white_mask, axis=0)
    below_cum = np.cumsum(white_mask[::-1, :], axis=0)[::-1, :]

    valid = np.zeros(H, dtype=bool)
    for y in range(H):
        row_nonwhite = ~white_mask[y, x0:x1+1]
        nnw = int(row_nonwhite.sum())
        if nnw < int(min_width_frac * width):
            continue

        has_above = above_cum[y-1, x0:x1+1] > 0 if y > 0   else np.zeros(width, bool)
        has_below = below_cum[y+1, x0:x1+1] > 0 if y < H-1 else np.zeros(width, bool)
        between_cols = row_nonwhite & has_above & has_below

        if nnw > 0 and (between_cols.sum() / float(nnw)) >= min_between_frac:
            valid[y] = True

    # longest contiguous run of valid rows
    if not valid.any():
        return np.zeros_like(white_mask, bool), None

    best_len, best_start = 0, 0
    i = 0
    while i < H:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < H and valid[j]:
            j += 1
        L = j - i
        if L > best_len:
            best_len, best_start = L, i
        i = j

    if best_len < min_band_rows:
        return np.zeros_like(white_mask, bool), None

    top, bot = best_start, best_start + best_len - 1
    band = np.zeros_like(white_mask, bool)
    band[top:bot+1, x0:x1+1] = True

    # small dilation to make the band solid (bridges 1-px gaps)
    band = cv2.dilate(band.astype(np.uint8), np.ones((3,3), np.uint8), 1).astype(bool)
    return band, (top, bot)


# --- helpers ---------------------------------------------------------------
def _moving_average(a, k=31):
    if k <= 1: return a
    k = int(max(1, k))
    pad = k // 2
    ap = np.pad(a, (pad, pad), mode='edge')
    return np.convolve(ap, np.ones(k)/k, mode='valid')

def _fill_nan_nearest(a, fill):
    x = np.arange(a.size)
    m = ~np.isnan(a)
    if m.sum() == 0:  return np.full_like(a, fill, float)
    if m.sum() == 1:  return np.full_like(a, a[m][0], float)
    return np.interp(x, x[m], a[m])

def _top_bottom_y_arrays(white_mask: np.ndarray):
    H, W = white_mask.shape
    has = white_mask.any(axis=0)

    top = np.full(W, np.nan, float)
    bot = np.full(W, np.nan, float)

    first = np.argmax(white_mask, axis=0)
    top[has] = first[has]
    rev_first = np.argmax(white_mask[::-1, :], axis=0)
    bot[has] = (H-1) - rev_first[has]

    top = _fill_nan_nearest(top, H/2)
    bot = _fill_nan_nearest(bot, H/2)
    return top, bot

# --- jaggedness of a 1D edge profile y(x) ---
def _top_bottom_y_in_roi(wm_roi: np.ndarray, H: int):
    """top_y, bot_y for a ROI (H x Wc) boolean mask."""
    has = wm_roi.any(axis=0)
    Wc = wm_roi.shape[1]
    top = np.full(Wc, np.nan, float)
    bot = np.full(Wc, np.nan, float)

    first = np.argmax(wm_roi, axis=0)
    top[has] = first[has]
    rev_first = np.argmax(wm_roi[::-1, :], axis=0)
    bot[has] = (H - 1) - rev_first[has]
    # fill tiny gaps so we can do math on all columns
    top = _fill_nan_nearest(top, H/2)
    bot = _fill_nan_nearest(bot, H/2)
    return top.astype(int), bot.astype(int)

def _max_contiguous_true(run_bool: np.ndarray) -> int:
    """length of the longest contiguous True run in a 1D bool array."""
    best = cur = 0
    for v in run_bool:
        if v:
            cur += 1
            if cur > best: best = cur
        else:
            cur = 0
    return best

def side_has_deep_jagged_runs(
    white_mask: np.ndarray,
    comp,                          # CompSpan
    side: str,                     # 'top' or 'bottom'
    min_run_cols: int = 1,         # “more than 3 contiguous” => ≥4 columns
    min_height_frac: float = 1/4.0 # height ≥ 1/3 component height
):
    """
    Implements your jaggedness rule for one side.

    For each column inside the component’s x-span:
      - build the vertical non-white run adjacent to that side,
      - keep only the portion of that run where the row still has some white
        in the ROI (so it's within the arch),
      - if its height ≥ (1/3)*component_height, the column 'qualifies'.

    Returns (qualifies_bool, qualified_columns_bool_array).
    """
    H, W = white_mask.shape
    x0, x1 = int(comp.getminx()), int(comp.getmaxx())
    x0 = max(0, min(x0, W-1)); x1 = max(0, min(x1, W-1))
    if x1 <= x0:
        return False, np.zeros(0, bool)

    wm_roi = white_mask[:, x0:x1+1]            # H x Wc
    row_has_white_in_roi = wm_roi.any(axis=1)  # H
    if not row_has_white_in_roi.any():
        return False, np.zeros(x1-x0+1, bool)

    # rows that are still within the arch (have some white in ROI)
    y_first = int(np.argmax(row_has_white_in_roi))
    y_last  = int((H - 1) - np.argmax(row_has_white_in_roi[::-1]))

    top_y, bot_y = _top_bottom_y_in_roi(wm_roi, H)
    Hc = max(1, int(comp.yspan()))
    thr_h = max(1, int(min_height_frac * Hc))

    if side == 'bottom':
        # vertical run from (bot_y[x]+1) down to y_last (inclusive), all rows have some white somewhere
        gap_h = np.maximum(0, y_last - bot_y)
    else:  # 'top'
        # vertical run from y_first up to (top_y[x]-1), inclusive
        gap_h = np.maximum(0, top_y - y_first)

    qualified_cols = (gap_h >= thr_h)
    qualifies = (_max_contiguous_true(qualified_cols) >= min_run_cols)
    return qualifies, qualified_cols

def _max_contiguous_true(run_bool: np.ndarray) -> int:
    best = cur = 0
    for v in run_bool:
        if v: cur += 1; best = max(best, cur)
        else: cur = 0
    return best

def _top_bottom_y_in_roi(wm_roi: np.ndarray):
    H, Wc = wm_roi.shape
    has = wm_roi.any(axis=0)
    top = np.full(Wc, np.nan, float)
    bot = np.full(Wc, np.nan, float)
    first = np.argmax(wm_roi, axis=0)
    top[has] = first[has]
    rev_first = np.argmax(wm_roi[::-1, :], axis=0)
    bot[has] = (H - 1) - rev_first[has]
    top = _fill_nan_nearest(top, H/2).astype(int)
    bot = _fill_nan_nearest(bot, H/2).astype(int)
    return top, bot

def hull_concavity_scores(white_mask, comp, band_frac=0.35, min_gap_h_frac=0.10):
    """
    Returns (#gap pixels near top, #gap pixels near bottom) measured on hull-mask
    within the component bbox. band_frac = portion of height used as top/bottom bands.
    Small gaps (height < min_gap_h_frac * comp height) are ignored.
    """
    _, gaps = hull_hollow_masks(white_mask)  # hull \ white (bool)

    y0, y1 = comp.getmin(), comp.getmax()
    x0, x1 = comp.getminx(), comp.getmaxx()
    g = gaps[y0:y1+1, x0:x1+1].astype(np.uint8)
    H, W = g.shape
    if H == 0 or W == 0:
        return 0, 0

    # Remove tiny speckle gaps and require a minimum vertical extent
    min_gap_h = max(3, int(min_gap_h_frac * H))
    nlab, lab = cv2.connectedComponents(g, connectivity=8)
    keep = np.zeros_like(g, dtype=np.uint8)
    for i in range(1, nlab):
        ys, xs = np.where(lab == i)
        if ys.size == 0: 
            continue
        if (ys.max() - ys.min() + 1) >= min_gap_h:
            keep[ys, xs] = 1
    g = keep.astype(bool)

    band = max(1, int(band_frac * H))
    top_band = np.zeros_like(g, bool);  top_band[:band, :]  = True
    bot_band = np.zeros_like(g, bool);  bot_band[-band:, :] = True

    return int(np.sum(g & top_band)), int(np.sum(g & bot_band))


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

    white_mask = binarize_mask(mask, use_otsu=True)   # or use_otsu=False, thr=128..200 if you prefer
    
# NEW: try to split by the row-bridge rule
# --- optional split by row-bridge rule (your function) ---
    white_mask_for_cc, split_y = split_mask_if_row_bridge(white_mask, width_frac=0.10, both_frac=0.50)
    if split_y is not None:
        print(f"[split] splitting on row y={split_y}")

# Components: adjacency only
    components = components_from_mask(white_mask_for_cc, connectivity=8, min_area_frac=0.01)

    print(f"Found {len(components)} components")

    non_white_mask = ~white_mask
    dist = cv2.distanceTransform((mask != 255).astype(np.uint8), cv2.DIST_L2, 3)
    far_from_white_mask = (dist > cutcount) & non_white_mask

# Start a collector for “extra” blackout (bands, bbox-gaps, etc.)
    extra_black = np.zeros_like(white_mask, bool)

# 1) Global between-band (independent of component count)
    band_mask1, band_rows1 = middle_band_by_row_test(
    white_mask, min_width_frac=0.07, min_between_frac=0.45, min_band_rows=5
    )
    if band_rows1 is not None:
        extra_black |= band_mask1

# 2) If we already have >=2 components, try a tighter band on their x-span; else fallback to bbox gap
    if len(components) >= 2:
        x0 = min(c.getminx() for c in components)
        x1 = max(c.getmaxx() for c in components)
        band_mask2, band_rows2 = middle_band_by_row_test(
        white_mask, min_width_frac=0.10, min_between_frac=0.50, min_band_rows=3, x0=x0, x1=x1
    )
        if band_rows2 is not None:
            extra_black |= band_mask2
        else:
        # bbox-y gap fallback
            gap_top = min(c.getmax() for c in components) + 1
            gap_bot = max(c.getmin() for c in components) - 1
            if gap_bot >= gap_top:
                gap_mask = np.zeros_like(white_mask, bool)
                gap_mask[gap_top:gap_bot+1, x0:x1+1] = True
                extra_black |= gap_mask

# ---------- Combine masks (default path for 2+ comps) ----------
    pixels_to_black_out = white_mask | far_from_white_mask | extra_black

# ---------- Single-component logic ----------
    if len(components) == 1:
        comp = components[0]
        if False:
            inner_centroid_hollow, all_hull_gaps = hull_hollow_masks(white_mask)
            inner_hollow = (inner_centroid_hollow | all_hull_gaps) & (dist <= cutcount)
            pixels_to_black_out = ~inner_hollow | extra_black   # <— keep band blackout
        else:
            near_bg = (~white_mask) & (dist <= cutcount)
            comp = components[0]
            top_cvx, bot_cvx = hull_concavity_scores(
    white_mask, comp, band_frac=0.35, min_gap_h_frac=0.10
)

            tau_area = max(50, int(0.002 * comp.xspan() * comp.yspan()))  # small safety floor
            close=0.2
            picked =None
            if max(top_cvx, bot_cvx) >= tau_area:
                if abs(top_cvx - bot_cvx) <= close * max(top_cvx, bot_cvx):
                    picked = 'both'
                else:
                    picked = 'top' if top_cvx > bot_cvx else 'bottom'

            if picked is None:
                picked = 'both'
    
                picked = decision  # 'top' | 'bottom' | 'both' | 'none'

# 3) build the mask to keep
            if picked == 'both' or picked == 'none':          # keep both when tie/none
                chosen_bg = near_bg
            elif picked == 'top':
                top_mask, _, _ = row_side_flatness_masks(white_mask)
                chosen_bg = near_bg & top_mask
            elif picked == 'bottom':
                _, bot_mask, _ = row_side_flatness_masks(white_mask)
                chosen_bg = near_bg & bot_mask

            pixels_to_black_out = ~chosen_bg | extra_black
            
    keep_count = np.sum(~pixels_to_black_out)
    print(f"Keeping {keep_count} pixels; cutcount={cutcount} (h={h}, w={w})")

    result = image.copy()
    result[pixels_to_black_out] = [0, 0, 0]

    # Save as PNG (forced by create_image_in_dir)
    create_image_in_dir(result, base_name)

    del image, mask, result, white_mask, dist, non_white_mask, far_from_white_mask
    gc.collect()


IMAGE_EXTS  = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ALLOWED_EXTS = IMAGE_EXTS

def find_mask_path(masks_dir: str, image_filename: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(image_filename))[0].lower()

    # 1) same name+ext as the image
    same_ext = os.path.join(masks_dir, os.path.basename(image_filename))
    if os.path.isfile(same_ext):
        return same_ext

    # 2) same stem, any allowed ext
    for ext in ALLOWED_EXTS:
        p = os.path.join(masks_dir, stem + ext)
        if os.path.isfile(p):
            return p

    # 3) common suffixes
    for ext in ALLOWED_EXTS:
        for suf in ("_mask", "-mask", "_m", "-m"):
            p = os.path.join(masks_dir, stem + suf + ext)
            if os.path.isfile(p):
                return p

    # 4) case-insensitive directory scan fallback
    for f in os.listdir(masks_dir):
        name_noext, ext = os.path.splitext(f)
        if ext.lower() in ALLOWED_EXTS:
            n = name_noext.lower()
            if n == stem or n == stem + "_mask" or n == stem + "-mask":
                return os.path.join(masks_dir, f)

    return None

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
        img_path = os.path.join(sys.argv[1], filename)
        mask_path = find_mask_path(sys.argv[2], filename)

        print(f"[{i}/{len(image_files)}] {filename}")
        if not (os.path.isfile(img_path) and mask_path and os.path.isfile(mask_path)):
            print(f"[!] Skipping {filename} - missing image or mask (tried: {mask_path})")
            continue

        try:
            cut_to_the_chase_efficient(img_path, mask_path)
            # or cut_to_the_chase_chunked(img_path, mask_path)
            print(f"[✓] Completed {filename}")
        except Exception as e:
            print(f"[✗] Error processing {filename}: {e}")

        gc.collect()

if __name__ == "__main__":

    main()
