import os
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from infer import load_trocr_model  # Import the function we made earlier

def run_single_test():
    # Load model & processor
    model, processor = load_trocr_model()
    model.eval()

    # Pick one image from processed data
    test_dir = "data/processed"
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not test_files:
        print("❌ No images found in data/processed")
        return

    img_path = os.path.join(test_dir, test_files[0])
    print(f"🖼 Testing with: {img_path}")

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Process and generate text
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("📝 Predicted text:", generated_text)

if __name__ == "__main__":
    run_single_test()
