import librosa
import numpy as np


def extract_advanced_features(signal, sr):

    features = {}

    # -------------------
    # MFCC
    # -------------------

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):

        features[f"mfcc_{i+1}"] = np.mean(
            mfcc[i]
        )

    # -------------------
    # Chroma
    # -------------------

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    for i in range(12):

        features[f"chroma_{i+1}"] = np.mean(
            chroma[i]
        )

    # -------------------
    # Spectral Contrast
    # -------------------

    contrast = librosa.feature.spectral_contrast(
        y=signal,
        sr=sr
    )

    for i in range(contrast.shape[0]):

        features[f"contrast_{i+1}"] = np.mean(
            contrast[i]
        )

    # -------------------
    # Tonnetz
    # -------------------

    harmonic = librosa.effects.harmonic(
        signal
    )

    tonnetz = librosa.feature.tonnetz(
        y=harmonic,
        sr=sr
    )

    for i in range(tonnetz.shape[0]):

        features[f"tonnetz_{i+1}"] = np.mean(
            tonnetz[i]
        )

    return features