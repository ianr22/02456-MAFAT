import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


# ===== CONFIG =====
train_csv = './dataset_v2/train.csv'
image_root = './dataset_v2/root/train'
output_root = './blurred_train_images_12'
os.makedirs(output_root, exist_ok=True)


# ===== SIGMA BLUR SETTINGS =====
num_levels = 12                 # blur_0 → blur_11
sigmas = np.linspace(0.0, 4.0, num_levels)


# ===== METADATA STORAGE =====
metadata_records = []


# ===== FUNCTIONS =====
def crop_polygon(image, coords):
    h, w = image.shape[:2]
    pts = np.array(coords, dtype=np.int32).reshape((4, 2))


    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)


    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)


    if x_max <= x_min or y_max <= y_min:
        return None


    cropped = image[y_min:y_max, x_min:x_max]
    pts_cropped = pts - np.array([x_min, y_min])


    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts_cropped], 255)
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    return result




# ===== MAIN =====
def main():
    df = pd.read_csv(train_csv)
    df.columns = df.columns.str.strip()


    coord_cols = ['p1_x', 'p_1y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y']
    for col in coord_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")


    # Pre-create blur folders
    for i in range(num_levels):
        os.makedirs(os.path.join(output_root, f"blur_{i}"), exist_ok=True)


    image_cache = {}


    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping and blurring images"):


        image_id = str(int(row['image_id']))
        tag_id = str(row['tag_id'])


        # ---------- Robust image loading ----------
        if image_id not in image_cache:
            loaded = False
            for ext in ['.jpg', '.jpeg', '.tif', '.tiff', '.png']:
                image_path = os.path.join(image_root, f"{image_id}{ext}")
                if not os.path.exists(image_path):
                    continue


                img = cv2.imread(image_path)
                if img is None:
                    try:
                        pil_img = Image.open(image_path).convert("RGB")
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except:
                        continue


                image_cache[image_id] = img
                loaded = True
                break


            if not loaded:
                print(f"⚠️ Could not load image {image_id}")
                continue


        image = image_cache[image_id]


        # ---------- Crop the vehicle ----------
        coords = row[coord_cols].astype(float).values
        cropped = crop_polygon(image, coords)
        if cropped is None or cropped.size == 0:
            continue


        # ---------- Apply sigma blur levels ----------
        for i, sigma in enumerate(sigmas):


            if sigma == 0:
                blurred = cropped
            else:
                blurred = cv2.GaussianBlur(cropped, (0, 0), sigmaX=sigma, sigmaY=sigma)


            blur_folder = os.path.join(output_root, f"blur_{i}")
            filename = f"{image_id}_{tag_id}.jpg"
            save_path = os.path.join(blur_folder, filename)


            success = cv2.imwrite(save_path, blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                print(f"⚠️ Failed to save {save_path}")
                continue


            # Add metadata record
            metadata_records.append({
                "image_id": image_id,
                "tag_id": tag_id,
                "filename": filename,
                "blur_level": i,
                "sigma": float(sigma),
                "output_path": save_path
            })


    # ===== Save metadata =====
    df_out = pd.DataFrame(metadata_records)
    df_out.to_csv("processed_blurred_metadata.csv", index=False)


    print("✅ Processing complete. Metadata saved to processed_blurred_metadata.csv")




# ===== RUN =====
if __name__ == "__main__":
    main()
