import torch
from models import PaLIGemmaClassifier, PaLIGemmaGPT
from transformers import AutoProcessor, GPT2Tokenizer
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classification model
classifier = PaLIGemmaClassifier("google/paligemma-2b", num_classes=6503).to(device)
classifier.load_state_dict(torch.load("classification_model.pt", map_location=device))
classifier.eval()

# Load generation model
generator = PaLIGemmaGPT("google/paligemma-2b", "gpt2").to(device)
generator.load_state_dict(torch.load("generation_model.pt", map_location=device))
generator.eval()

processor = AutoProcessor.from_pretrained("google/paligemma-2b")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Provide mapping (simplified)
idx2ans = {0: "yes", 1: "no", 2: "unanswerable", 3: "red", 4: "blue", 5: "shirt"}  # etc.
ans2idx = {v: k for k, v in idx2ans.items()}

def hybrid_answer(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = classifier(inputs["pixel_values"], inputs["input_ids"])
        pred_idx = torch.argmax(logits, dim=-1).item()
        pred_ans = idx2ans.get(pred_idx, "unanswerable")

    if pred_ans == "unanswerable":
        gen_input = tokenizer([f"Question: {question} Answer:"], return_tensors="pt").to(device)
        with torch.no_grad():
            output = generator(inputs["pixel_values"], gen_input.input_ids)
            generated_ids = torch.argmax(output.logits, dim=-1)
            pred_ans = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return pred_ans.strip()
