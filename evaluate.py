import torch
from torch.utils.data import DataLoader
from data_loader import VQADataset
from models import PaLIGemmaClassifier, PaLIGemmaGPT
from transformers import AutoProcessor, GPT2Tokenizer
from utils import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_classification(test_csv, img_dir, ans2idx, idx2ans):
    dataset = VQADataset(test_csv, img_dir, "google/paligemma-2b", mode="classification", answer2idx=ans2idx)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = PaLIGemmaClassifier("google/paligemma-2b", num_classes=len(ans2idx)).to(device)
    model.load_state_dict(torch.load("classification_model.pt", map_location=device))
    model.eval()

    preds, refs = [], []
    with torch.no_grad():
        for pixel_values, input_ids, labels in dataloader:
            pixel_values, input_ids = pixel_values.to(device), input_ids.to(device)
            logits = model(pixel_values, input_ids)
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend([idx2ans[p] for p in pred])
            refs.extend([idx2ans[l.item()] for l in labels])

    metrics = compute_metrics(preds, refs)
    print("Classification Metrics:", metrics)

def evaluate_generation(test_csv, img_dir):
    dataset = VQADataset(test_csv, img_dir, "google/paligemma-2b", mode="generation")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = PaLIGemmaGPT("google/paligemma-2b", "gpt2").to(device)
    model.load_state_dict(torch.load("generation_model.pt", map_location=device))
    model.eval()
    processor = AutoProcessor.from_pretrained("google/paligemma-2b")

    preds, refs = [], []
    with torch.no_grad():
        for pixel_values, question, answer in dataloader:
            pixel_values = pixel_values.to(device)
            inputs = tokenizer([f"Question: {question[0]} Answer:"], return_tensors="pt").to(device)
            outputs = model(pixel_values, inputs.input_ids)
            generated_ids = torch.argmax(outputs.logits, dim=-1)
            pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            preds.append(pred_text.strip())
            refs.append(answer[0].strip())

    metrics = compute_metrics(preds, refs)
    print("Generation Metrics:", metrics)
