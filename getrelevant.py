import cv2
import os
import sys
import numpy as np
import gc

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
    if not np.any(white_mask):
        print(f"Warning: No white pixels found in {mask_path}")
        return

    # Distance from each NON-WHITE pixel to the nearest WHITE pixel (Euclidean)
    # Foreground = non-white (1), background = white (0)
    non_white_u8 = (mask != 255).astype(np.uint8)
    dist = cv2.distanceTransform(non_white_u8, cv2.DIST_L2, 3)

    # Pixels farther than the radius are "far from white"
    non_white_mask = (mask != 255)
    far_from_white_mask = (dist > cutcount) & non_white_mask

    # Black out all white pixels and all far non-white pixels
    pixels_to_black_out = white_mask | far_from_white_mask

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
    output_filename = f"{name}_relevant{ext}"
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

