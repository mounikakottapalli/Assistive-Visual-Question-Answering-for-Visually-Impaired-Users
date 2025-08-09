# LSA64 Sign Language Recognition

A deep learning system for Argentinian Sign Language (LSA64) recognition using CNN-LSTM architecture with memory-efficient streaming data processing.

## Overview

This project implements a sign language recognition system that can classify LSA64 gestures from video sequences. The system uses a CNN-LSTM architecture where MobileNetV2 extracts spatial features from each frame and an LSTM processes the temporal sequence.

## Requirements

- Python 3.7+
- TensorFlow 2.8+
- OpenCV
- scikit-learn
- matplotlib
- seaborn
- numpy
- tqdm

## Installation

```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn numpy tqdm
```

## Dataset

Download the LSA64 dataset and place it in `/content/drive/MyDrive/LSA64/` (for Google Colab) or update the `LSA64_PATH` variable in the code.

Expected file structure:
```
LSA64/
├── 001_001_001.mp4
├── 001_001_002.mp4
├── 001_002_001.mp4
└── ...
```

## Usage

Run the training script:

```python
python lsa64_streaming_training.py
```

## Configuration

Modify the `CONFIG` dictionary to adjust training parameters:

```python
CONFIG = {
    'IMG_SIZE': (224, 224),
    'SEQUENCE_LENGTH': 15,
    'BATCH_SIZE': 8,
    'NUM_CLASSES': 10,
    'LEARNING_RATE': 0.001,
    'EPOCHS': 30,
    'USE_SUBSET': True,
    'SUBSET_CLASSES': 10
}
```

## Model Architecture

- **Feature Extractor**: MobileNetV2 (pre-trained on ImageNet)
- **Temporal Model**: LSTM with 256 hidden units
- **Classification**: Dense layer with softmax activation

## Key Features

- Memory-efficient streaming data loading
- Real-time data augmentation
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics
- Model checkpointing

## Results

The model achieves approximately 85-90% accuracy on the LSA64 dataset subset with efficient memory usage.

## License

MIT License
