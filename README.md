
# EmotiVoice AI — Speech Emotion Recognition

EmotiVoice AI is a deep learning project that estimates emotion categories from speech audio using MFCC features and a CNN-LSTM model.

## Features

- WAV audio upload
- MFCC feature extraction
- CNN-LSTM speech emotion classifier
- Emotion predictions and model confidence scores
- Flask web interface

## Emotion classes

Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.

## Technology

Python, Flask, TensorFlow/Keras, Librosa, NumPy, Scikit-learn, HTML, CSS and JavaScript.

## Dataset

This project uses the RAVDESS speech audio dataset.
Dataset: https://zenodo.org/records/1188976

Please follow the dataset's attribution and non-commercial/share-alike licensing terms.

## Run locally

1. Install Python.
2. Install dependencies: `pip install -r requirements.txt`
3. Download and extract the RAVDESS speech dataset.
4. Run `python train.py` to train the model.
5. Run `python app.py` to start the website.
6. Open http://127.0.0.1:5000

The trained model files must exist in the `models/` directory before predictions will work.

## Disclaimer

The model estimates emotion expressed in a recording. It cannot reliably determine a person's true internal emotional state.
