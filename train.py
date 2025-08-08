import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from data_loader import VQADataset
from models import PaLIGemmaClassifier, PaLIGemmaGPT
from transformers import GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classification(ans2idx, train_csv, img_dir):
    dataset = VQADataset(train_csv, img_dir, "google/paligemma-2b", mode="classification", answer2idx=ans2idx)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = PaLIGemmaClassifier("google/paligemma-2b", num_classes=len(ans2idx)).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for pixel_values, input_ids, labels in dataloader:
            pixel_values, input_ids, labels = pixel_values.to(device), input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pixel_values, input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "classification_model.pt")

def train_generation(train_csv, img_dir):
    dataset = VQADataset(train_csv, img_dir, "google/paligemma-2b", mode="generation")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = PaLIGemmaGPT("google/paligemma-2b", "gpt2").to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for pixel_values, question, answer in dataloader:
            pixel_values = pixel_values.to(device)
            inputs = tokenizer([f"Question: {q} Answer:" for q in question], return_tensors="pt", padding=True).to(device)
            labels = tokenizer(list(answer), return_tensors="pt", padding=True).input_ids.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values, inputs.input_ids, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "generation_model.pt")
