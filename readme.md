# Hand Gesture Recognition System

This project is designed for recognizing hand gestures using a hybrid deep learning model. It processes hand landmarks and angles extracted from webcam frames or pre-recorded datasets.

---

## Datasets
The dataset required for training called data_add can be downloaded from the following URL:

[Download Dataset](https://liveuclac-my.sharepoint.com/:u:/g/personal/zcabnnu_ucl_ac_uk/EU7kBJPk9L5BharkPhkzwGYB0XOC1mCyKiCMxtQE-wtLeA?e=xMScQc)


The dataset required for evaluation called data can be downloaded from the following URL:

[Download Dataset](https://liveuclac-my.sharepoint.com/:u:/g/personal/zcabnnu_ucl_ac_uk/Ed99fsTjxJtCpLqIWnd7oCoBP1XOEuAkcTAKf9NvREkcsA?e=aVWbWr)



## How to Use the Dataset

1. **Download the Dataset**:
   - Click on the Download Dataset link to download the dataset.

2. **Extract the Dataset**:
   - Extract the downloaded dataset into the root directory.

3. **Verify the Structure**:
   - Ensure the dataset structure matches the one specified above.

## Components

The repository structure is organized as follows:

- **`data`**: Contains images used for training the model.
- **`data_add`**: Contains images used for validating the model.
- **`dataset_collection`**: Provides details on the data collection process.
- **`model`**: Includes code to train a model using only hand landmarks as input.
- **`model_angles`**: Extends the model with calculated angles as additional input features alongside hand landmarks.
- **`model_hybrid`**: Contains the primary hybrid model combining both angles and hand landmarks, used throughout this project.
- **`validation`**: Contains scripts to evaluate model performance, including metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
- **`verification`**: Includes scripts to run real-time model tests locally using your device’s webcam.
- **`conversion.py`**: Script to convert the trained Keras model into TensorFlow Lite (TFLite) format for deployment within the mobile application.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # For Unix/MacOS
.\venv\Scripts\activate   # For Windows

pip install -r requirements.txt
```