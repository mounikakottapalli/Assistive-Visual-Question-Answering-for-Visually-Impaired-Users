import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score
import numpy as np

nltk.download('wordnet')
nltk.download('omw-1.4')

def compute_metrics(preds, refs):
    bleu_scores = []
    meteor_scores = []
    correct = 0

    for p, r in zip(preds, refs):
        bleu_scores.append(sentence_bleu([r.split()], p.split(), smoothing_function=SmoothingFunction().method1))
        meteor_scores.append(meteor_score([r], p))
        if p.strip().lower() == r.strip().lower():
            correct += 1

    acc = correct / len(preds)
    f1 = f1_score([r.lower() for r in refs], [p.lower() for p in preds], average='micro')

    return {
        "accuracy": acc,
        "bleu": np.mean(bleu_scores),
        "meteor": np.mean(meteor_scores),
        "f1": f1
    }
