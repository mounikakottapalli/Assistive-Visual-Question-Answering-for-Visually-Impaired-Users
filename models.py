import torch
import torch.nn as nn
from transformers import AutoModel, GPT2LMHeadModel

class PaLIGemmaClassifier(nn.Module):
    def __init__(self, encoder_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids):
        outputs = self.encoder(pixel_values=pixel_values, input_ids=input_ids)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

class PaLIGemmaGPT(nn.Module):
    def __init__(self, encoder_name, gpt_name="gpt2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, self.gpt.config.hidden_size)

    def forward(self, pixel_values, input_ids, labels=None):
        vision_outputs = self.encoder(pixel_values=pixel_values)
        vision_embeds = self.proj(vision_outputs.last_hidden_state[:, 0, :])
        inputs_embeds = self.gpt.transformer.wte(input_ids)
        fused = torch.cat((vision_embeds.unsqueeze(1), inputs_embeds), dim=1)
        return self.gpt(inputs_embeds=fused, labels=labels)
