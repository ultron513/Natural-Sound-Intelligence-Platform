from fastapi import (
    FastAPI,
    UploadFile,
    File
)
from pathlib import Path
from backend.routes.prediction import router as prediction_router
from app.predictor import predict_sound
import shutil

from app.config import (
    DATASET_DIR,
    UPLOADS_DIR
)
from backend.routes.health import (
    router as health_router
)
from backend.config import (
    API_TITLE,
    API_VERSION
)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION
)
app.include_router(
    prediction_router,
    tags=["Prediction"]
)
app.include_router(
    health_router,
    tags=["Health"]
)

UPLOADS_DIR.mkdir(
    exist_ok=True
)

@app.get("/")
def home():

    return {
        "message": "Natural Sound Classification API Running",
        "model": "Random Forest",
        "classes": 13
    }


from backend.config import *

@app.get("/info")
def model_info():

    return {

        "project":
        API_TITLE,

        "model":
        MODEL_NAME,

        "features":
        TOTAL_FEATURES,

        "classes":
        TOTAL_CLASSES,

        "test_accuracy":
        TEST_ACCURACY,

        "full_dataset_accuracy":
        FULL_DATASET_ACCURACY
    }
@app.get("/predict-test")
def predict_test():

    audio_path = (
        DATASET_DIR /
        "ESC-50-master" /
        "audio"
    )

    test_file = list(
        audio_path.glob("*.wav")
    )[0]

    result = predict_sound(
        test_file
    )

    return result