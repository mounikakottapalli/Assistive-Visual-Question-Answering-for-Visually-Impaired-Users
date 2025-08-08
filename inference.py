from transformers import AutoProcessor, GPT2Tokenizer
from models import PaLIGemmaClassifier, PaLIGemmaGPT
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classify_image(image_path, question, model, processor, idx2ans):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt")
    logits = model(inputs["pixel_values"].to(device), inputs["input_ids"].to(device))
    pred = torch.argmax(logits, dim=-1).item()
    return idx2ans[pred]

def generate_answer(image_path, question, model, processor, tokenizer):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    input_ids = tokenizer.encode(f"Question: {question} Answer:", return_tensors="pt")
    outputs = model(inputs["pixel_values"].to(device), input_ids.to(device))
    generated = tokenizer.decode(torch.argmax(outputs.logits, dim=-1)[0])
    return generated

if __name__ == "__main__":
    print("Run classification or generation using pre-trained models")
