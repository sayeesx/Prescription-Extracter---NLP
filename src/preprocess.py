import cv2
import os
import numpy as np
from pathlib import Path

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] == 0:  # no text pixels found
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(img_path, save_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not load image: {img_path}")
        return

    # Thresholding (binarization)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew
    deskewed = deskew(thresh)

    # Remove noise
    denoised = cv2.medianBlur(deskewed, 3)

    cv2.imwrite(save_path, denoised)

def run_preprocessing():
    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(RAW_DIR, filename)
            out_path = os.path.join(OUT_DIR, filename)
            preprocess_image(in_path, out_path)
            print(f"✅ Processed {filename}")

if __name__ == "__main__":
    run_preprocessing()
