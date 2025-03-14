import gdown
import os
import kaggle

# URL of the Google Drive folder
url = 'https://drive.google.com/drive/u/1/folders/1gLYXB_krVpUDv6Qcc932OjjOCe7R4J4c'

# Extract the folder ID from the URL
folder_id = url.split('/')[-1]

# Create the checkpoints folder if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# Download the files from the Google Drive folder to the checkpoints folder
gdown.download_folder(f'https://drive.google.com/drive/folders/{folder_id}?usp=sharing', output='checkpoints', quiet=False, use_cookies=False)

# Kaggle dataset URL
dataset_url = 'abdelghaniaaba/wildfire-prediction-dataset'

# Create the checkpoints folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the dataset from Kaggle
kaggle.api.dataset_download_files(dataset_url, path='data', unzip=True)
