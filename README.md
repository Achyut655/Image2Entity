# Visual Information Extractor

A machine learning system that extracts product entity values from images using computer vision and natural language processing techniques. The system processes 250,000 images and achieves a 70% F1 score by combining OpenCV, EasyOCR, and BERT transformers.

## Overview

This project implements a two-stage pipeline:
1. Text Extraction: Uses computer vision techniques to preprocess images and extract text
2. Entity Value Prediction: Employs BERT-based models to predict entity values from extracted text

## Features

- Image preprocessing with OpenCV
- Automatic text orientation detection and correction
- Text extraction using EasyOCR
- BERT-based sequence classification
- Support for batch processing of images
- Checkpoint system for training recovery

## Requirements

### Dependencies
```
python>=3.7
torch>=1.8.0
transformers>=4.5.0
opencv-python>=4.5.0
easyocr>=1.4.0
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
pillow>=8.0.0
tqdm>=4.60.0
```

### Hardware Requirements
- GPU with CUDA support (recommended)
- Minimum 16GB RAM

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/visual-information-extractor.git
cd visual-information-extractor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Text Extraction
```python
from src.text_extraction.ocr_pipeline import process_images_from_csv

# Process images and extract text
process_images_from_csv(
    csv_file='path/to/input.csv',
    output_file='path/to/output.csv'
)
```

### 2. Training the Model
```python
# Train the BERT classifier
python src/model/training.py \
    --input_file processed_data.csv \
    --epochs 25 \
    --batch_size 16
```

### 3. Making Predictions
```python
from src.model.bert_classifier import predict_entity_value

# Predict entity value for extracted text
text = "Product weight: 500g"
entity_name = "item_weight"
predicted_value = predict_entity_value(text, entity_name)
```

## Model Performance

- Dataset size: 250,000 images
- F1 Score: 70%
- Accuracy improvement: 15%
- Training time: ~8 hours on V100 GPU
- Inference time: ~0.5 seconds per image

## Future Improvements

- Implement data augmentation techniques
- Add support for multiple languages
- Optimize inference speed
- Add ensemble models

