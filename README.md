# Visual Information Extractor

A machine learning system that extracts product entity values from images using computer vision and natural language processing techniques. The system processes 250,000 images and achieves a 70% F1 score by combining OpenCV, EasyOCR, and BERT transformers.

## Overview

This project implements a two-stage pipeline:
1. Text Extraction: Uses computer vision techniques to preprocess images and extract text
2. Entity Value Prediction: Employs BERT-based models to predict entity values from extracted text

## Project Structure

```
Image2Entity/
├── image_extraction/
│   └── extract.ipynb          # OCR pipeline for text extraction
├── model_train/
│   └── model.ipynb            # BERT-based model training
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## Features

- Image preprocessing with OpenCV
- Automatic text orientation detection and correction
- Text extraction using EasyOCR
- BERT-based sequence classification
- Support for batch processing of images
- Checkpoint system for training recovery

```

### Hardware Requirements
- GPU with CUDA support (recommended)
- Minimum 16GB RAM

## Installation

1. Clone the repository
```bash
git clone https://github.com/Achyut655/Image2Entity.git
cd Image2Entity
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Text Extraction
```python
# Run the Jupyter notebook for text extraction
# Open image_extraction/extract.ipynb in Jupyter or Google Colab
# The notebook contains the complete OCR pipeline for processing images

# Example usage from the notebook:
csv_file = 'path/to/your/input.csv'  # Replace with your CSV file path
output_file = 'output_with_extracted_text.csv'  # Output file
process_images_from_csv(csv_file, output_file)
```

### 2. Training the Model
```python
# Run the Jupyter notebook for model training
# Open model_train/model.ipynb in Jupyter or Google Colab
# The notebook contains the complete BERT-based training pipeline

# The model will be saved as:
# - entity_value_predictor_model/
# - entity_value_predictor_tokenizer/
```

### 3. Making Predictions
```python
# The prediction function is included in the model training notebook
# Example usage:
text = "Product weight: 500g"
entity_name = "item_weight"
predicted_value = predict_entity_value(text, entity_name)
print(f"Predicted entity value: {predicted_value}")
```

## Model Performance

- Dataset size: 250,000 images
- F1 Score: 70%
- Accuracy improvement: 15%
- Training time: ~8 hours on V100 GPU
- Inference time: ~0.5 seconds per image

## Data Format

The notebooks expect the following data format:

### For Text Extraction (`extract.ipynb`):
- Input CSV should contain an `image_link` column with URLs to images
- Output CSV will include an `extracted_text` column with the extracted text

### For Model Training (`model.ipynb`):
- Input CSV should contain:
  - `extracted_text`: Text extracted from images
  - `entity_name`: Name of the entity to predict
  - `entity_value`: Target value for the entity

## Future Improvements

- Implement data augmentation techniques
- Add support for multiple languages
- Optimize inference speed
- Add ensemble models

