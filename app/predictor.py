import joblib
import librosa
import numpy as np
import pandas as pd

from app.config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH
)

from app.feature_extractor import (
    extract_advanced_features
)


# ----------------------------
# Load Once
# ----------------------------

model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

encoder = joblib.load(
    ENCODER_PATH
)


# ----------------------------
# Prediction Function
# ----------------------------

def predict_sound(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=None
    )

    features = extract_advanced_features(
        signal,
        sr
    )

    feature_df = pd.DataFrame(
        [features]
    )

    scaled = scaler.transform(
        feature_df
    )

    prediction = model.predict(
        scaled
    )

    probabilities = model.predict_proba(
        scaled
    )

    predicted_class = encoder.inverse_transform(
        prediction
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    class_names = encoder.classes_

    top_indices = np.argsort(
        probabilities[0]
    )[::-1][:3]
    
    top_predictions = []
    
    for idx in top_indices:
    
        top_predictions.append(
            {
                "class_name": str(
                    class_names[idx]
                ),
                "probability": round(
                    float(
                        probabilities[0][idx]
                    ) * 100,
                    2
                )
            }
        )
    
    return {
        "prediction": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        ),
        "top_predictions": top_predictions
    }