import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm


# ===== CONFIG =====
test_csv = './dataset_v2/test.csv'
image_root = './dataset_v2/root/test'
output_root = './blurred_test_images'
kernel_sizes = [0, 11, 21, 31, 41]   # 0 = original
level_names = ['Original', 'Blur1', 'Blur2', 'Blur3', 'Blur4']


# ===== FUNCTIONS =====
def crop_polygon(image, coords):
    """
    Crop an image based on 8 polygon coordinates (4 points).
    Returns cropped region with alpha mask applied to polygon shape.
    """
    h, w = image.shape[:2]
    pts = np.array(coords, dtype=np.int32).reshape((4, 2))
   
    # Compute bounding box for polygon
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)


    # Clip to image dimensions
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)


    if x_max <= x_min or y_max <= y_min:
        return None


    # Crop region of interest
    cropped = image[y_min:y_max, x_min:x_max]
   
    # Adjust polygon coordinates relative to crop
    pts_cropped = pts - np.array([x_min, y_min])
   
    # Create mask for polygon
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts_cropped], 255)
   
    # Apply mask to crop
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    return result


# ===== MAIN FUNCTION =====
def main():
    df = pd.read_csv(test_csv)
    df.columns = df.columns.str.strip()  # clean up whitespace


    # Correct coordinate column names
    coord_cols = ['p1_x', 'p_1y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y']
    for col in coord_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")


    # Ensure output directories exist
    os.makedirs(output_root, exist_ok=True)
    for name in level_names:
        os.makedirs(os.path.join(output_root, name), exist_ok=True)


    # Cache loaded images (for images with multiple annotations)
    image_cache = {}


    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping & blurring test objects"):
        image_id = str(int(row['image_id']))
        tag_id = str(row['tag_id']) if 'tag_id' in df.columns else f"{idx}"


        # Try both jpg and tiff
        image_path_jpg = os.path.join(image_root, f"{image_id}.jpg")
        image_path_tiff = os.path.join(image_root, f"{image_id}.tiff")


        if image_id not in image_cache:
            if os.path.exists(image_path_jpg):
                image = cv2.imread(image_path_jpg)
            elif os.path.exists(image_path_tiff):
                image = cv2.imread(image_path_tiff)
            else:
                print(f"⚠️ Image not found for ID {image_id}")
                continue
            image_cache[image_id] = image
        else:
            image = image_cache[image_id]


        if image is None:
            continue


        # Extract polygon coordinates
        coords = row[coord_cols].astype(float).values
        cropped = crop_polygon(image, coords)
        if cropped is None or cropped.size == 0:
            continue


        # Apply blur levels
        for ksize, name in zip(kernel_sizes, level_names):
            if ksize == 0:
                blurred = cropped
            else:
                blurred = cv2.GaussianBlur(cropped, (ksize, ksize), 0)


            # Save crop per tag
            out_filename = f"{image_id}_{tag_id}.png"
            out_path = os.path.join(output_root, name, out_filename)
            cv2.imwrite(out_path, blurred)


    print("\n✅ All test objects processed and saved with blur levels in:", output_root)


# ===== RUN =====
if __name__ == "__main__":
    main()


