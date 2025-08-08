import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import AutoProcessor

class VQADataset(Dataset):
    def __init__(self, csv_path, image_dir, processor_name, mode="classification", answer2idx=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.mode = mode
        self.answer2idx = answer2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.image_dir}/{self.data.iloc[idx]['image']}"
        question = self.data.iloc[idx]['question']
        answer = self.data.iloc[idx]['answer']

        image = Image.open(img_path).convert("RGB")
        processed = self.processor(images=image, text=question, return_tensors="pt", padding=True)

        if self.mode == "classification":
            label = self.answer2idx.get(answer, self.answer2idx.get("unanswerable"))
            return processed["pixel_values"].squeeze(0), processed["input_ids"].squeeze(0), torch.tensor(label)
        else:
            return processed["pixel_values"].squeeze(0), question, answer
