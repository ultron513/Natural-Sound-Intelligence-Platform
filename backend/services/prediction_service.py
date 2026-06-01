from app.predictor import predict_sound


def predict_audio_file(

    audio_path

):

    result = predict_sound(
        audio_path
    )

    return result