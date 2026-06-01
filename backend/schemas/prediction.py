from pydantic import BaseModel


class TopPrediction(
    BaseModel
):

    class_name: str

    probability: float


class PredictionResponse(
    BaseModel
):

    prediction: str

    confidence: float

    top_predictions: list[TopPrediction]