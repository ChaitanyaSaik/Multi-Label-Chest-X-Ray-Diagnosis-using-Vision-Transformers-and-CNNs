
# Multi-Label Chest X-Ray Diagnosis using Vision Transformers and CNNs

A deep learning research project comparing Vision Transformers (ViT) and DenseNet121 CNN for multi-label chest disease classification using the NIH ChestX-ray14 dataset.

---

## Project Overview

Chest X-rays are widely used for diagnosing thoracic diseases such as pneumonia, cardiomegaly, and effusion. However, manual interpretation is time‑consuming and subject to inter‑observer variability. This project builds an AI-based computer-aided diagnosis system to detect multiple thoracic diseases simultaneously.

---

## Problem Statement

Chest X-ray analysis has several challenges:

1. Multi-label classification – one image can contain multiple diseases.
2. Severe class imbalance – some diseases are rare.
3. Global context understanding – abnormalities may spread across lung regions.
4. Threshold calibration – fixed thresholds reduce performance for rare classes.

The goal is to develop a deep learning system that improves detection performance while handling these challenges.

---

## Research Gap

Existing research has limitations:

- CNN vs Transformer comparisons often use different training pipelines.
- Class imbalance handling is frequently ignored.
- Fixed thresholds (0.5) reduce performance for minority diseases.
- Many works lack explainability.
- Most studies focus on binary classification instead of multi‑label diagnosis.

This project addresses these gaps through a controlled comparison of CNN and transformer architectures.

---

## Dataset

NIH ChestX-ray14 Dataset

Total Images: 112,120  
Unique Patients: 30,805  
Disease Classes: 14

Dataset Link:
https://nihcc.app.box.com/v/ChestXray-NIHCC

---

## Data Preprocessing

- Image resize to 224x224
- ImageNet normalization
- Data augmentation:
  - horizontal flip
  - small rotation
  - contrast adjustment

Dataset split:

Training: 70%  
Validation: 15%  
Testing: 15%

Patient-wise splitting prevents data leakage.

---

## Models

### Vision Transformer (ViT)

- Uses self-attention to capture global dependencies
- Better at modeling distributed abnormalities across the lung

### DenseNet121

- CNN architecture with dense connections
- Efficient feature reuse and strong baseline performance

---

## Training Strategy

Optimizer: AdamW

Loss Function: Class Weighted Binary Cross Entropy

Training method:

1. Train classification head
2. Progressive fine-tuning of full network
3. Early stopping using validation Micro-AUC

---

## Threshold Optimization

Instead of a fixed threshold (0.5), the model uses per-class threshold tuning to improve performance for minority diseases.

---

## Explainability

Grad-CAM is used for CNN models to visualize disease-relevant regions.

Vision Transformer attention maps help visualize which areas of the X-ray influence predictions.

---

## Experimental Results

Model Performance

DenseNet121  
Accuracy: 85.6%  
Micro AUC: 69.05%  
Micro F1: 18.93%

Vision Transformer  
Accuracy: 87.22%  
Micro AUC: 77.40%  
Micro F1: 25.23%

Vision Transformer performs better in capturing global patterns in chest radiographs.

---

## Technologies Used

Python  
PyTorch  
Torchvision  
NumPy  
Pandas  
Scikit-learn  
Matplotlib

Models:

Vision Transformer (ViT)  
DenseNet121

---

## Project Structure
'''
project/
│
├── dataset/
├── models/
│   ├── vit_model.py
│   └── densenet_model.py
├── training/
│   └── train.py
├── explainability/
│   └── gradcam.py
├── notebooks/
│   └── experiments.ipynb
├── paper/
│   └── Final_Paper.pdf
├── requirements.txt
└── README.md
'''
---

## Future Work

- Federated learning for privacy‑preserving medical AI
- Improve rare disease detection
- Self-supervised pretraining
- Clinical validation

---

## Authors

Undergraduate Researchers  
Department of Computer Science and Engineering

Srivatsal Parise  
Gidda Neha Devaki  
Kurapati Chaitanya Sai  
Ratnala Navaneeth  

Supervisor  
Dr. V Lakshmi Chetana
