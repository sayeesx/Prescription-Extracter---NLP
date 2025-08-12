import cv2
import os
import numpy as np
from pathlib import Path

PROCESSED_DIR = "data/processed"
LINES_DIR = "data/lines"
Path(LINES_DIR).mkdir(parents=True, exist_ok=True)

def segment_lines(image_path, base_name):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return
    
    # Invert colors (text = white, bg = black)
    inv_img = cv2.bitwise_not(img)

    # Sum pixels horizontally
    hist = np.sum(inv_img, axis=1)

    # Threshold for detecting blank rows
    threshold = np.max(hist) * 0.05

    # Identify line start/end
    lines = []
    start = None
    for i, val in enumerate(hist):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            end = i
            if end - start > 10:  # ignore tiny lines
                lines.append((start, end))
            start = None

    # Save cropped lines
    count = 1
    for (y1, y2) in lines:
        line_img = img[y1:y2, :]
        out_name = f"{base_name}_line{count}.jpg"
        cv2.imwrite(os.path.join(LINES_DIR, out_name), line_img)
        count += 1

def run_segmentation():
    for filename in os.listdir(PROCESSED_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(PROCESSED_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            segment_lines(in_path, base_name)
            print(f"✅ Segmented {filename}")

if __name__ == "__main__":
    run_segmentation()
