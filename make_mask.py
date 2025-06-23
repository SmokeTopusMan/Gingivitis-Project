import sys
import os
from PIL import Image, ImageDraw

def rename_images_in_directory(directory_path, start_number=1):
    # Supported image&text extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.txt'}

    # Get list of image files sorted alphabetically
    image_files = sorted([
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f.lower())[1] in image_extensions
    ])

    # Rename files
    for i, filename in enumerate(image_files, start=start_number):
        old_path = os.path.join(directory_path, filename)
        extension = os.path.splitext(filename)[1]
        new_filename = f"{i}{extension}"
        new_path = os.path.join(directory_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")


def convert_non_white_to_black(path):
    # Load image
    img = Image.open(path).convert('RGB')
    pixels = img.load()

    width, height = img.size

    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            if (r, g, b) != (255, 255, 255):
                pixels[x, y] = (0, 0, 0)

    img.save(path)
    print(f"Saved processed image to: {path}")


def yolo_polygon_to_mask(txt_path, output_path, img_size):
    """
    Converts YOLO polygon annotation to a binary mask image.

    Args:
        txt_path (str): Path to the .txt annotation file.
        output_path (str): Path to save the output binary mask image.
        img_size (tuple): (width, height) of the target mask image.
    """

    img_width, img_height = img_size
    mask = Image.new("L", (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    with open(txt_path, "r") as f:
        for line in f:
            tokens = list(map(float, line.strip().split()))
            if len(tokens) < 3:
                continue
            polygon_points = tokens[1:]
            points = [(int(x * img_width), int(y * img_height))
                      for x, y in zip(polygon_points[::2], polygon_points[1::2])]
            draw.polygon(points, outline=255, fill=255)

    mask.save(output_path)


def yolo_generate_masks(image_dir, label_dir, output_dir):
    """
    Generates binary masks for all images in a directory using corresponding YOLO labels.

    Args:
        image_dir (str): Path to directory containing images.
        label_dir (str): Path to directory containing YOLO polygon `.txt` files.
        output_dir (str): Path to save the binary mask images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_filename in os.listdir(image_dir):
        if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        name, _ = os.path.splitext(image_filename)
        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(label_dir, f"{name}.txt")
        output_path = os.path.join(output_dir, f"{name}.jpg")

        if not os.path.exists(label_path):
            print(f"[!] No label file for {image_filename}")
            continue

        with Image.open(image_path) as img:
            img_size = img.size  # (width, height)

        yolo_polygon_to_mask(label_path, output_path, img_size)
        print(f"[✓] Created mask for {image_filename}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python make_mask.py directory")
        sys.exit(1)

    if not os.path.isdir(sys.argv[1]):
        print(f"Error: {sys.argv[1]} is not a valid directory.")
        sys.exit(1)

    image_dir = os.path.join(sys.argv[1], "images")
    label_dir = os.path.join(sys.argv[1], "labels")
    output_dir = os.path.join(sys.argv[1], "masks")
    rename_images_in_directory(image_dir, 404)
    rename_images_in_directory(label_dir, 404)
    yolo_generate_masks(image_dir, label_dir, output_dir)


if __name__ == "__main__":
    main()


