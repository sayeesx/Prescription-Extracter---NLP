import os
from transformers import VisionEncoderDecoderModel

def load_trocr_model():
    local_model_path = "./models/trocr"

    try:
        if os.path.exists(local_model_path):
            print("✅ Loading local big model...")
            model = VisionEncoderDecoderModel.from_pretrained(local_model_path)
            processor = TrOCRProcessor.from_pretrained(local_model_path)
        else:
            print("⬇ Downloading big model (microsoft/trocr-base-handwritten)... This may take a while...")
            model_name = "microsoft/trocr-base-handwritten"
            model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=local_model_path)
            processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=local_model_path)
    except KeyboardInterrupt:
        print("\n⚠ Download interrupted. Using small fallback model...")
        model_name = "microsoft/trocr-small-handwritten"
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = TrOCRProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"⚠ Error loading big model: {e}")
        print("Using small fallback model...")
        model_name = "microsoft/trocr-small-handwritten"
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        processor = TrOCRProcessor.from_pretrained(model_name)

    return model, processor

if __name__ == "__main__":
    model, processor = load_trocr_model()
    print("✅ Model loaded successfully!")
