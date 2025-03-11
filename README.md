# Project Wildfire: Wildfire Detection

This repository contains the implementation of a wildfire detection system using machine learning techniques. The project leverages a dataset from Kaggle to train models for predicting wildfire occurrences.

## Dataset

The dataset used in this project is available on Kaggle: [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset). The dataset contains three folders: a training set, a validation set, and a test set. However, specific constraints and guidelines are followed:

1. **Dataset Access and Restrictions:** The dataset is composed of a *training set*, a *validation set*, and a *test set*. The project does not use the labels of the training set.

2. **Dataset Splitting:** The original validation set is split into a new validation set and a new train set.

## Project Structure

The repository is organized as follows:

```
.
├── baseline_vit-finetune_swin-finetune
├── checkpoints
├── coloration_baseline
├── context_encoder
├── data
├── dino_vit_pretrain
├── download.py
├── outputs
├── README.md
├── requirements.txt
└── Self_training
```

- **baseline_vit-finetune_swin-finetune/**: Contains the scripts for finetuning all the models on the validation dataset.
  - `models.py`: Definitions of the models used for finetuning.
  - `train.py`: Script to train the models.
  - `utils.py`: Utility functions for training and evaluation.

- **checkpoints/**: Directory to store model checkpoints.
  - **coloration_baseline/**: Checkpoints for the coloration baseline model.
  - **context_encoder/**: Checkpoints for the context encoder model.
  - **dino_backbone/**: Checkpoints for the DINO backbone model.
  - **dino_vit_finetuned/**: Checkpoints for the finetuned DINO Vision Transformer (ViT) model.
  - **satlas_swin_finetuned/**: Checkpoints for the finetuned Satlas Swin Transformer model.
  - **Self-training (pseudo-labeling)/**: Checkpoints for the self-training (pseudo-labeling) model.

- **coloration_baseline/**: Contains the scripts for the coloration baseline model.
  - `dataset.py`: Dataset loading and preprocessing.
  - `main.py`: Main script to run the coloration baseline model.
  - `model.py`: Definition of the coloration baseline model.
  - `train.py`: Script to train the coloration baseline model.
  - `utils.py`: Utility functions for training and evaluation.

- **context_encoder/**: Contains the main scripts and utilities for training and evaluating the context encoder model.
  - `classifier.py`: Implementation of the classifier model.
  - `model.py`: Definition of the context encoder model.
  - `train_classifier.py`: Script to train the classifier model.
  - `train_context_encoder.py`: Script to train the context encoder model.
  - `utils/`: Utility scripts for loading datasets and other helper functions.
    - `load_dataset.py`: Script to load the dataset.
    - `utils_classifier.py`: Utility functions for the classifier.
    - `utils_gan.py`: Utility functions for the GAN.
  - `visualisation.ipynb`: Jupyter notebook for visualizing model outputs and data.

- **data/**: Directory to store dataset files.
  - **test/**: Test dataset.
  - **train/**: Training dataset.
  - **valid/**: Validation dataset.

- **dino_vit_pretrain/**: Contains the scripts for pre-training the DINO model.
  - `train_dino_backbone.py`: Script to train the DINO backbone model.
  - `utils_dino.py`: Utility functions for training the DINO model.
  - `vision_transformer.py`: Definition of the Vision Transformer (ViT) model.

- **outputs/**: Directory to store output files such as visualizations and plots.
  - **context_encoder/**: Outputs related to the context encoder model.
    - `context_encoder_False_training_plot.png`: Training plot for the context encoder model with False setting.
    - `context_encoder__training_plot.png`: Training plot for the context encoder model.
    - `context_encoder_True_training_plot.png`: Training plot for the context encoder model with True setting.
    - `reconstructions_masked.png`: Visualization of masked reconstructions.
    - `reconstructions_masked_zoomed.png`: Zoomed visualization of masked reconstructions.
    - `reconstructions.png`: Visualization of reconstructions.
    - `reconstructions_zoomed.png`: Zoomed visualization of reconstructions.

- **Self_training/**: Contains the scripts for self-training (pseudo-labeling).
  - `baselineCNN.py`: Definition of the baseline CNN model.
  - `pseudolabel_finetune.ipynb`: Jupyter notebook for pseudo-labeling and finetuning.
  - `test_models.ipynb`: Jupyter notebook for testing models.
  - `train_baseline.ipynb`: Jupyter notebook for training the baseline model.

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

3. **Download the dataset and weights:**
   ```bash
   python download.py
   ```
   Note: Weights are available in this [drive](https://drive.google.com/drive/folders/1gLYXB_krVpUDv6Qcc932OjjOCe7R4J4c?usp=drive_link)

4. **Download Pretrained Swin Transformer (Optional):**
   If training the **Swin Transformer**, download the pretrained weights:
   ```bash
   wget https://huggingface.co/allenai/satlas-pretrain/resolve/main/satlas-model-v1-highres.pth -O satlas-model-v1-highres.pth
   ```

5. **Train the models:**

   - **Context Encoder:**
     1. Train the context encoder model and classifier:
       ```bash
       cd context_encoder/
       python train_context_encoder.py
       ```
     2. Train the classifier model:
       ```bash       
       cd context_encoder/
       python train_classifier.py
       ```
     3. Visualize the results:
        Open the `visualisation.ipynb` notebook to visualize the model outputs and data.

   - **Coloration Baseline:**
     ```bash
     cd coloration_baseline/
     python main.py
     ```

   - **DINO ViT Pretrain:**
     ```bash
     cd dino_vit_pretrain/
     python train_dino_backbone.py
     ```

   - **Finetune Models:**
     ```bash
     cd baseline_vit-finetune_swin-finetune/
     python train.py --model model_to_fine_tune[vit,swin,baseline]
     ```

## Acknowledgments

- Pre-train with Context Encoders: [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/pdf/1604.07379)
- Swin pretrained transformed [Open ai-generated geospatial data](https://github.com/allenai/satlas?tab=readme-ov-file)
- Dino algorithm and vit implementation [DINO](https://github.com/facebookresearch/dino)
