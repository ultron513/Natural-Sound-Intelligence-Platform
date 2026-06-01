from fastapi import (
    APIRouter,
    UploadFile,
    File
)
from backend.schemas.prediction import (
    PredictionResponse
)
import shutil

from app.predictor import predict_sound
from app.config import UPLOADS_DIR
from backend.services.prediction_service import (
    predict_audio_file
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=
    PredictionResponse
)
async def predict_uploaded_file(

    file: UploadFile = File(...)

):

    save_path = (
        UPLOADS_DIR /
        file.filename
    )

    with open(
        save_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_audio_file(
    save_path
    )

    return result