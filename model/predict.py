import numpy as np
import time
import joblib
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D,
    Concatenate, BatchNormalization,
    Bidirectional, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D
)

test_file_path = r"test_npy_samples\sample_8_label_0.npy"

def build_feature_extractor():

    input_layer = Input(shape=(2548,1))

    cnn1 = Conv1D(128,3,activation='relu',padding='same')(input_layer)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPooling1D(2)(cnn1)

    cnn2 = Conv1D(128,5,activation='relu',padding='same')(input_layer)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPooling1D(2)(cnn2)

    cnn3 = Conv1D(128,7,activation='relu',padding='same')(input_layer)
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = MaxPooling1D(2)(cnn3)

    cnn4 = Conv1D(128,11,activation='relu',padding='same')(input_layer)
    cnn4 = BatchNormalization()(cnn4)
    cnn4 = MaxPooling1D(2)(cnn4)

    merged = Concatenate()([cnn1,cnn2,cnn3,cnn4])

    attn_output = MultiHeadAttention(num_heads=4,key_dim=64)(merged,merged)
    attn_output = LayerNormalization()(attn_output + merged)

    bilstm = Bidirectional(LSTM(256,return_sequences=True))(attn_output)

    gap = GlobalAveragePooling1D()(bilstm)

    feature_vector = Dense(256,activation='relu')(gap)

    model = Model(inputs=input_layer, outputs=feature_vector)

    return model

print("Loading models...")

feature_extractor = build_feature_extractor()
feature_extractor.load_weights("saved_models/feature_extractor_model.h5")

xgb_model = joblib.load("saved_models/xgboost_classifier.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")

print("Models loaded successfully")


sample = np.load(test_file_path)
print("Loaded sample shape:", sample.shape)

start = time.time()


sample_scaled = scaler.transform(sample.reshape(1,-1))
sample_scaled = sample_scaled.reshape(1,2548,1)

deep_features = feature_extractor.predict(sample_scaled, verbose=0)

probs = xgb_model.predict_proba(deep_features)

pred_idx = np.argmax(probs)
confidence = probs[0][pred_idx]

predicted_class = label_encoder.inverse_transform([pred_idx])[0]

end = time.time()

print("\nPrediction Result")
print("--------------------------------")
print("Predicted Emotion :", predicted_class)
print("Confidence        :", round(confidence*100,2), "%")
print("Prediction Time   :", round(end-start,4), "seconds")