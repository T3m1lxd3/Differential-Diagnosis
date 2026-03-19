# Differential Diagnosis with Explainable AI

An AI/ML system that predicts disease diagnoses from patient symptoms and explains its predictions using Explainable AI (XAI) techniques.

## Overview

This project tackles the challenge of building a transparent medical diagnosis model. Given a patient's symptoms and vitals, the system predicts the most likely disease from a set of 55+ conditions, and critically, it explains *why* it made that prediction using SHAP and LIME, making the model interpretable for real-world medical contexts.

## Features

- Multi-class disease classification across 55+ conditions
- Two ML approaches compared: Random Forest and Neural Network
- Class imbalance handling via **SMOTE** (Synthetic Minority Oversampling Technique)
- Hyperparameter tuning with **GridSearchCV**
- Model explainability with **SHAP** and **LIME**
- Feature importance analysis

## Dataset

- **Source**: `disease_symptoms.csv` — 349 patient records across 8 features
- **Features**: Fever, Cough, Fatigue, Difficulty Breathing, Age, Gender, Blood Pressure, Cholesterol Level
- **Target**: Disease label (116 unique classes before filtering, 55 after removing low-sample classes)

## Models & Results

### Random Forest (baseline)
- **Without SMOTE**: 31% accuracy — severely impacted by class imbalance
- **With SMOTE**: **85% accuracy** — SMOTE dramatically improved performance by balancing underrepresented disease classes

### Neural Network (TensorFlow/Keras)
- Architecture: `Dense(256) → Dropout(0.2) → Dense(128) → Dropout(0.2) → Softmax`
- Optimizer: Adam (lr=0.001) with ReduceLROnPlateau scheduler
- **Test accuracy: 78.3%** over 50 epochs

### Key Finding
Random Forest with SMOTE outperformed the Neural Network on this dataset, likely due to the small dataset size (349 records) limiting the neural network's ability to generalize.

## Explainability

A major focus of this project is making the model's decisions transparent:

- **SHAP (SHapley Additive exPlanations)**: Used KernelExplainer to compute feature contributions for neural network predictions across all 55 disease classes
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explains individual predictions by approximating the model locally — showing which symptoms most influenced a specific diagnosis

Example: For a patient with `Fever=Yes, Fatigue=Yes, Difficulty Breathing=Yes, Age=19`, the model predicts a diagnosis and LIME highlights which symptoms drove that classification.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

**Key libraries**: `tensorflow`, `scikit-learn`, `imbalanced-learn`, `shap`, `lime`, `seaborn`, `matplotlib`

## Project Structure

```
Differential-Diagnosis/
├── AI Project Notebook.ipynb   # Full pipeline: EDA → preprocessing → modeling → XAI
├── disease_symptoms.csv        # Primary dataset
└── symbipredict_2022.csv       # Supporting dataset
```

## How to Run

```bash
# Install dependencies
pip install tensorflow scikit-learn imbalanced-learn shap lime pandas seaborn matplotlib

# Open the notebook
jupyter notebook "AI Project Notebook.ipynb"
```

Run all cells in order. The notebook is self-contained and walks through each stage of the pipeline.

## Topics

`machine-learning` `explainable-ai` `healthcare` `random-forest` `neural-network` `shap` `lime` `classification` `python` `jupyter-notebook`
