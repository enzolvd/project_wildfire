# Project Wildfire: Wildfire Detection

This repository contains the implementation of a wildfire detection system using machine learning techniques. The project leverages a dataset from Kaggle to train models for predicting wildfire occurrences.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)

## Introduction

Wildfires pose a significant threat to ecosystems and human communities. Early detection and prediction of wildfires can help mitigate their impact. This project aims to develop a robust system for wildfire detection using deep learning models.

## Dataset

The dataset used in this project is available on Kaggle: [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset). It contains various features that are relevant to wildfire prediction, such as weather conditions, geographical information, and historical wildfire data.

## Project Structure

The repository is organized as follows:

```
.
├── baseline
├── checkpoints
│   └── baseline
│   └── context_encoder
│   └── satlas_swin
├── context_encoder
├── data
├── satlas_swin
├── outputs
└── README.md
```

- **checkpoints/**: Contains saved model checkpoints.
- **context_encoder/**: Contains the main scripts and utilities for training and evaluating the context encoder model.
  - `classifier.py`: Implementation of the classifier model.
  - `model.py`: Definition of the context encoder model.
  - `train_classifier.py`: Script to train the classifier model.
  - `train_context_encoder.py`: Script to train the context encoder model.
  - `utils/`: Utility scripts for loading datasets and other helper functions.
  - `visualisation.ipynb`: Jupyter notebook for visualizing model outputs and data.
- **data/**: Directory to store dataset files.
- **outputs/**: Directory to store model outputs and results.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/enzolvd/project_wildfire.git
   cd project_wildfire
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) and place it in the `data/` directory.
  
4. **Train the models:**
   - Train the context encoder model:
     ```bash
     python context_encoder/train_context_encoder.py
     ```
   - Train the classifier model:
     ```bash
     python context_encoder/train_classifier.py
     ```

5. **Visualize the results:**
   - Open the `visualisation.ipynb` notebook to visualize the model outputs and data.
  
6. **Models weight**
   - Weights are available at [model weights](https://shorturl.at/JuJes)
