"""
Command-line audio classifier using a pretrained CNN model

Usage example:
-------------
python src/inference/predict.py --audio data/samples/street_music.wav

"""

import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from src.utils.audio_utils import extract_feature


MODEL_PATH = "saved_models/cnn_model_after_training.h5"
FEATURE_PATH = "data/features/features_from_UrbanSound_for_cnn.h5"

def load_label_encoder(feature_path: str):
    """Loads label encoder from feature dataframe."""
    df = pd.read_hdf(feature_path, "df")
    le = LabelEncoder()
    le.fit(df.class_label.tolist())
    return le

def predict_audio(model_path: str, feature_path: str, audio_path: str):
    """Predicts class for a single audio file."""
    if not os.path.exists(model_path):
        print("Model not found! Run 'python src/models/train_cnn.py' first.")
        return
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    model = load_model(model_path)
    le = load_label_encoder(feature_path)

    feat = extract_feature(audio_path)
    if feat is None:
        return

    feat = feat[np.newaxis, ..., np.newaxis]
    preds = model.predict(feat)
    idx = np.argmax(preds, axis=-1)
    pred_class = le.inverse_transform(idx)[0]

    print(f"Predicted class: {pred_class}")
    return pred_class
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict the sound of class using a pretrained CNN model"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        default= "data/samples/gunshot.wav"
        help= "Path to the input audio file to classfiy (.wav)"
    )
    
    args = parser.parse_args()
    
    predict_audio(MODEL_PATH, FEATURE_PATH, args.audio)
	