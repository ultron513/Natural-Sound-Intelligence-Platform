import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=None
    )

    plt.figure(figsize=(12,4))

    librosa.display.waveshow(
        signal,
        sr=sr
    )

    plt.title("Waveform")

    plt.tight_layout()

    plt.show()


def plot_spectrogram(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=None
    )

    spectrogram = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(signal)
        ),
        ref=np.max
    )

    plt.figure(figsize=(12,5))

    librosa.display.specshow(
        spectrogram,
        sr=sr,
        x_axis="time",
        y_axis="log"
    )

    plt.colorbar()

    plt.title("Spectrogram")

    plt.tight_layout()

    plt.show()