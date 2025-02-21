# Project Wildfire: Wildfire Detection

This repository contains the implementation of a wildfire detection system using machine learning techniques. The project leverages a dataset from Kaggle to train models for predicting wildfire occurrences.

## Dataset

The dataset used in this project is available on Kaggle: [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset).  The dataset contains three folders: a training set, a validation set, and a test set. However, there specific constraints and guidelines are followed:

  **1. Dataset Access and Restrictions:** the dataset is composed of a *training set*, a *validation set*, and a *test set*. The project does not use the labels of the training set

  **2. Dataset Splitting:**  the original validation set is split into a new validation set and a new train set. 



## Project Structure

The repository is organized as follows:

```
.
├── baseline_vit-finetune_swin-finetune
├── dino_vit_pretrain
├── context_encoder
├── data
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
  

## Acknowledgments

- Pre-train with Context Encoders: [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/pdf/1604.07379)
