# Assistive Visual Question Answering (VQA) System

This project implements an Assistive VQA System designed for visually impaired users. It supports:

- Classification-based VQA using PaLI-Gemma-2 + MLP
- Generation-based VQA using PaLI-Gemma-2 + GPT-2

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Format

CSV with:
```
image,question,answer
img1.jpg,What color is the shirt?,blue
```

## Train

### Classification
```bash
python train.py --mode classification
```

### Generation
```bash
python train.py --mode generation
```

## Evaluate

```bash
python evaluate.py --mode classification
python evaluate.py --mode generation
```

## Inference

```bash
python inference.py
```
