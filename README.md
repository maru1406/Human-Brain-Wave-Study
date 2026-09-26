@"
# Human Brain Wave Study for Emotion Recognition

An EEG-based emotion recognition system that combines deep learning and machine learning to extract meaningful features from human brain-wave signals and classify emotional states through a Django web application.

## Project Overview

This project analyzes EEG (Electroencephalography) signals for emotion recognition. A multi-scale 1D Convolutional Neural Network (CNN) is used to learn deep representations from EEG signals. The extracted features are then classified using an XGBoost classifier.

The trained models are integrated into a Django-based web application where users can submit EEG samples and receive a predicted emotion along with the model's confidence score.

## Methodology

```text
EEG Signal
    ↓
Preprocessing & Scaling
    ↓
Multi-Scale 1D CNN
(Kernel Sizes: 3, 5, 7, 11)
    ↓
Feature Extraction
    ↓
256-Dimensional Feature Representation
    ↓
XGBoost Classifier
    ↓
Emotion Prediction
    ↓
Predicted Emotion + Confidence Score

## Key Features

- EEG signal-based emotion recognition
- Multi-scale 1D CNN for deep feature extraction
- XGBoost-based emotion classification
- Feature scaling and label encoding
- Prediction confidence estimation
- Django-based web application
- User authentication and dashboard
- Prediction history management
- Integration of trained deep learning and machine learning models

## Model Architecture

The emotion recognition pipeline uses a multi-scale 1D Convolutional Neural Network (CNN) to learn meaningful representations from EEG signals.

The CNN uses parallel convolutional branches with different kernel sizes:

- Kernel Size 3
- Kernel Size 5
- Kernel Size 7
- Kernel Size 11

These branches capture patterns at different temporal scales in the EEG signal. The extracted features are combined and transformed into a **256-dimensional feature representation**.

The resulting features are passed to an **XGBoost classifier**, which predicts the emotional class. A label encoder is then used to map the predicted class to its corresponding emotion label.

```text
EEG Input
    ↓
Multi-Scale 1D CNN
    ├── Conv1D (3)
    ├── Conv1D (5)
    ├── Conv1D (7)
    └── Conv1D (11)
    ↓
Feature Combination
    ↓
256-D Feature Vector
    ↓
XGBoost Classifier
    ↓
Emotion Class
    ↓
Confidence Score

## Technologies Used

### Programming
- Python

### Machine Learning & Deep Learning
- TensorFlow
- Keras
- XGBoost
- Scikit-learn
- NumPy

### Web Development
- Django
- HTML
- CSS

### Model & Data Processing
- Jupyter Notebook
- H5
- Pickle

## Project Structure

```text
Human-Brain-Wave-Study/
│
├── accounts/
│   ├── forms.py
│   ├── models.py
│   ├── signals.py
│   └── views.py
│
├── dashboard/
│   ├── ml_model.py
│   ├── models.py
│   └── views.py
│
├── model/
│   ├── saved_models/
│   │   ├── deep_feature_model.h5
│   │   ├── feature_extractor_model.h5
│   │   ├── label_encoder.pkl
│   │   ├── scaler.pkl
│   │   └── xgboost_classifier.pkl
│   ├── model.ipynb
│   ├── predict.py
│   └── dataset link.txt
│
├── static/
├── templates/
├── ultimate_ai/
├── manage.py
├── requirements.txt
└── README.md

## Dataset

The project uses EEG (Electroencephalography) signal data for emotion recognition.

The dataset contains EEG samples that are processed and transformed into suitable input representations for the deep learning and machine learning pipeline.

The original dataset is not included in this repository due to its size. The dataset source is provided in:

`model/dataset link.txt`

### Data Processing

- EEG samples are loaded and preprocessed using NumPy.
- Input signals are scaled before being passed to the neural network.
- The processed EEG signals are provided to the multi-scale 1D CNN.
- Deep features are extracted from the trained neural network.
- The extracted features are classified using XGBoost.
## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/maru1406/Human-Brain-Wave-Study.git
cd Human-Brain-Wave-Study
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

## Prediction Output

For each EEG sample, the system processes the signal through the trained feature extraction and classification pipeline.

The application provides:

- Predicted emotion/class
- Prediction confidence score

The prediction workflow is:

```text
EEG Sample
    ↓
Preprocessing & Scaling
    ↓
Deep Feature Extraction
    ↓
XGBoost Classification
    ↓
Label Decoding
    ↓
Predicted Emotion + Confidence

## Future Improvements

- Explore advanced deep learning architectures for EEG-based emotion recognition
- Experiment with transformer-based models for EEG signal analysis
- Improve feature extraction and classification performance
- Expand the range of emotion classes
- Add interactive EEG signal visualizations
- Deploy the application on a cloud platform
- Support additional EEG datasets

## License

This project is intended for educational and research purposes.
