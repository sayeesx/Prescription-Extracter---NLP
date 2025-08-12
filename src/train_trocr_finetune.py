import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

# -------- CONFIG --------
MODEL_NAME = "./models/trocr"   # local or huggingface model path
TRAIN_CSV = "data/train/final_labels.csv"   # your training CSV path
TRAIN_IMG_DIR = "data/train"                   # folder with train images
EPOCHS = 3
BATCH_SIZE = 4
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- DATASET --------
class OCRDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['file_name'])  # use 'file_name' column
        image = Image.open(img_path).convert("RGB")
        # Use ground truth text column 'text' for training
        text = row['text']

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, padding="max_length", truncation=True, max_length=128).input_ids
        labels = torch.tensor(labels)

        return {"pixel_values": pixel_values, "labels": labels}

# -------- TRAIN FUNCTION --------
def train():
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    model.to(DEVICE)
    print(f"Training on device: {DEVICE}")

    dataset = OCRDataset(TRAIN_CSV, TRAIN_IMG_DIR, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, batch in enumerate(loop):
            print(f"Batch {i+1}/{len(dataloader)}")
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    output_dir = "./models/trocr-finetuned"
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    train()
