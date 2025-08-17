# HumanEmotions: Facial Expression Recognition with Custom CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)

A custom Convolutional Neural Network (CNN) for facial expression recognition on the FER-2013 (kaggle) dataset, enabling emotion prediction (angry, disgust, fear, happy, neutral, sad, surprise) from image uploads. The project includes a training pipeline, a Streamlit web app for interactive predictions, and a FastAPI-based RESTful API for integration with diverse applications, leveraging 18+ years of IT integration expertise in HR and Telecom.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Streamlit App](#streamlit-app)
  - [FastAPI](#fastapi)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The **HumanEmotions** project implements a custom CNN (`FER_CNN`) in PyTorch for facial expression recognition using the FER-2013 dataset. It predicts seven emotions from grayscale 48x48 images. The project includes:

- A training script to build and save the model (`HumanEmotionsModel/HumanEmotions.pth`).
- A Streamlit app (`HumanEmotionsStreamlit/humanemotions.py`) for uploading images and viewing predictions with confidence scores.
- A FastAPI app (`HumanEmotionsFastAPI/HumanEmotions.py`) for a RESTful API to integrate emotion prediction into other applications.

This project demonstrates skills in deep learning, computer vision, model deployment, and API development, tailored for applications like HR sentiment analysis.

## Features

- **Custom CNN**: A 4-layer CNN with batch normalization and dropout for robust emotion classification.
- **Data Pipeline**: Handles FER-2013’s class imbalance with weighted sampling and aggressive augmentation (rotation, flips, cropping).
- **Interactive App**: Streamlit interface for uploading images and visualizing predictions with confidence score bar charts.
- **API Integration**: FastAPI endpoint (`/predictEmotions`) for scalable emotion prediction in web/mobile/HR systems.
- **Resume-Ready**: Showcases end-to-end ML development, from data preprocessing to production deployment.

## Dataset

The project uses the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), containing ~35,000 grayscale 48x48 images across 7 emotion classes:

- Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Organized in `data/train` and `data/test` folders with subfolders for each emotion.

**Note**: The dataset is not included in this repository due to size. Download it from Kaggle and place it in a `data/` directory.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vinaygandhigit/HumanEmotions.git
   cd HumanEmotions
   ```

Install dependencies:pip install torch torchvision pillow numpy streamlit fastapi uvicorn scikit-learn seaborn matplotlib

Ensure HumanEmotions.pth (trained model) is in the root directory. If not, train the model using the training script (see Usage).
(Optional) Download the FER-2013 dataset from Kaggle and extract to data/.

Usage
Training the Model

Ensure the FER-2013 dataset is in data/train and data/test.
Run the training script HumanSentiment.ipynb

The trained model is saved as HumanEmotions.pth.

Streamlit App

Run the Streamlit app:streamlit run humanemotions.py

Open http://localhost:8501 in a browser.
Upload an image (JPG/JPEG) to predict the emotion and view confidence scores.

Sample Output:

Predicted Emotion: Happy
Confidence Scores: { "happy": 0.85, "neutral": 0.10, ... }

FastAPI

Run the FastAPI app:uvicorn fer_fastapi_app:app --reload

Open http://127.0.0.1:8000/docs for the Swagger UI.
Use the /predict endpoint to upload an image and get JSON results.

Sample cURL Command:
curl -X POST "http://127.0.0.1:8000/predictEmotions" -F "file=@path/to/image.jpg"

Sample Python Request:
import requests
files = {'file': open('path/to/image.jpg', 'rb')}
response = requests.post('http://127.0.0.1:8000/predictEmotions', files=files)
print(response.json())

Sample Response:
{
"predicted_emotion": "happy",
"confidence_scores": {
"angry": 0.05,
"disgust": 0.02,
"fear": 0.03,
"happy": 0.85,
"neutral": 0.03,
"sad": 0.01,
"surprise": 0.01
}
}

Model Architecture
The custom FER_CNN model is designed for grayscale 48x48 images:

Input: 1 channel, 48x48 pixels
Conv Layers:
Conv2d(1→64, 3x3, padding=1) → ReLU → BatchNorm → MaxPool(2x2)
Conv2d(64→128, 3x3, padding=1) → ReLU → BatchNorm → MaxPool(2x2)
Conv2d(128→256, 3x3, padding=1) → ReLU → BatchNorm → MaxPool(2x2)
Conv2d(256→512, 3x3, padding=1) → ReLU → BatchNorm → MaxPool(2x2)

Fully Connected Layers:
Linear(51233→512) → ReLU → Dropout(0.5)
Linear(512→256) → ReLU → Dropout(0.5)
Linear(256→7)

Output: 7 emotion classes

Performance

Dataset: FER-2013 (~28k train, ~7k test images)
Metrics: (Note: Current metrics show underfitting; re-training recommended)
Test Accuracy: ~60.29% (target: >60%)
Train Loss: ~1.0869 (target: <1.0)
Val Loss: ~1.0567 (target: <1.0)

Improvements: Currenty the model trained on 20 epochs, Increase epochs to 20-50.

Deployment

Local: Run Streamlit (streamlit run humanemotions.py) or FastAPI (uvicorn HumanEmotions:app).
Cloud: Deploy FastAPI to AWS, Heroku, or Render for production. Use Docker:FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn torch torchvision pillow numpy
CMD ["uvicorn", "HumanEmotions:app", "--host", "0.0.0.0", "--port", "8000"]

Streamlit Cloud: Push to GitHub and deploy humanemotions.py via Streamlit Cloud.

Future Improvements

Model: Switch to ResNet-18 for higher accuracy (70-75%).
Face Detection: Add OpenCV Haar cascades for robust face cropping.
Real-Time: Integrate webcam support in Streamlit using streamlit-webrtc.
Batch Processing: Extend FastAPI to handle multiple image uploads.
Hyperparameter Tuning: Optimize learning rate, batch size, and augmentations.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact

Author: Vinay Gandhi
GitHub: vinaygandhigit
