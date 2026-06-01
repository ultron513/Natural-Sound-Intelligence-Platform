import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
CATEGORY_MAP = {
    "water_drops": "💧 Water Drops",
    "chirping_birds": "🐦 Chirping Birds",
    "crackling_fire": "🔥 Crackling Fire",
    "crickets": "🦗 Crickets",
    "frog": "🐸 Frog",
    "rain": "🌧️ Rain",
    "rooster": "🐓 Rooster",
    "sea_waves": "🌊 Sea Waves",
    "thunderstorm": "⛈️ Thunderstorm",
    "wind": "💨 Wind",
    "crow": "🐦 Crow",
    "pouring_water": "🚰 Pouring Water",
    "insects": "🐞 Insects"
}

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Natural Sound Intelligence Platform",
    page_icon="🎵",
    layout="wide"
)
st.markdown(
    """
<style>

.main {
    padding-top: 1rem;
}

h1 {
    text-align: center;
}

div[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 15px;
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
}

div[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.1);
}

</style>
""",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Project Information")

st.sidebar.markdown("""
### Model
Random Forest Classifier

### Dataset
ESC-50

### Technologies
- FastAPI
- Streamlit
- Librosa
- Scikit-Learn
- Random Forest

### Performance
- Test Accuracy: 72.12%
- Dataset Accuracy: 94.42%
""")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    """
<div style="
padding:25px;
border-radius:15px;
background:linear-gradient(
90deg,
rgba(0,120,255,0.25),
rgba(0,200,120,0.15)
);
margin-bottom:20px;
">

<h1 style="margin-bottom:0px;">
🎵 Natural Sound Intelligence Platform
</h1>

<h3 style="margin-top:5px;">
Environmental Sound Classification Using Machine Learning
</h3>

<p>
Transform raw environmental audio into intelligent predictions using
advanced audio signal processing and machine learning.
</p>

</div>
""",
    unsafe_allow_html=True
)

hero1, hero2, hero3, hero4 = st.columns(4)
st.divider()

st.subheader(
    "🚦 System Status"
)

status1, status2, status3 = st.columns(3)

with status1:

    st.success(
        "Backend API Online"
    )

with status2:

    st.success(
        "Model Loaded"
    )

with status3:

    st.success(
        "Prediction Service Ready"
    )
with hero1:
    st.metric(
        "Audio Samples",
        "520"
    )

with hero2:
    st.metric(
        "Sound Classes",
        "13"
    )

with hero3:
    st.metric(
        "Features",
        "38"
    )

with hero4:
    st.metric(
        "Best Model",
        "RF"
    )

# ---------------------------------------------------
# BACKEND STATUS CHECK
# ---------------------------------------------------

try:

    response = requests.get(
        "http://127.0.0.1:8000/health"
    )

    if response.status_code == 200:

        st.success(
            "Backend Connected Successfully"
        )

    else:

        st.error(
            "Backend Connection Failed"
        )

except:

    st.error(
        "FastAPI Backend Not Running"
    )

# ---------------------------------------------------
# PROJECT STATS
# ---------------------------------------------------

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Classes",
        "13"
    )

with col2:
    st.metric(
        "Features",
        "38"
    )

with col3:
    st.metric(
        "Dataset Accuracy",
        "94.42%"
    )

st.divider()

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if uploaded_file is not None:

    st.subheader("Uploaded Audio")

    st.audio(uploaded_file)
    st.info(
        f"File Name: {uploaded_file.name}"
    )
    # ---------------------------------------------------
    # AUDIO ANALYTICS
    # ---------------------------------------------------
    audio_bytes = uploaded_file.read()
    with open(
        "temp_audio.wav",
        "wb"
    ) as f:
        f.write(audio_bytes)
    y, sr = librosa.load(
        "temp_audio.wav",
        sr=None
    )
    duration = round(
        librosa.get_duration(
            y=y,
            sr=sr
        ),
        2
    )
    audio_col1, audio_col2 = st.columns(2)
    with audio_col1:
        st.metric(
            "Duration (sec)",
            duration
        )
    with audio_col2:
        st.metric(
            "Sample Rate",
            sr
        )
    st.subheader(
        "🎵 Waveform"
    )
    fig_wave, ax_wave = plt.subplots(
        figsize=(10, 3)
    )
    librosa.display.waveshow(
        y,
        sr=sr,
        ax=ax_wave
    )
    ax_wave.set_title(
        "Audio Waveform"
    )
    st.pyplot(fig_wave)
    # ---------------------------------------------------
    # MEL SPECTROGRAM
    # ---------------------------------------------------
    st.subheader(
        "🎼 Mel Spectrogram"
    )

    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr
    )

    spectrogram_db = librosa.power_to_db(
        spectrogram,
        ref=np.max
    )

    fig_spec, ax_spec = plt.subplots(
        figsize=(10, 4)
    )

    img = librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax_spec
    )

    ax_spec.set_title(
        "Mel Spectrogram"
    )

    fig_spec.colorbar(
        img,
        ax=ax_spec,
        format="%+2.0f dB"
    )

    st.pyplot(fig_spec)
    # ---------------------------------------------------
    # AUDIO FEATURE ANALYSIS
    # ---------------------------------------------------

    st.subheader(
        "🔬 Audio Analysis"
    )

    rms_energy = float(
        np.mean(
            librosa.feature.rms(
                y=y
            )
        )
    )

    zero_crossing_rate = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y
            )
        )
    )

    spectral_centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )
    )

    analysis1, analysis2, analysis3 = st.columns(3)
    
    with analysis1:
    
        st.metric(
            "RMS Energy",
            f"{rms_energy:.4f}"
        )
    
    with analysis2:
    
        st.metric(
            "Zero Crossing Rate",
            f"{zero_crossing_rate:.4f}"
        )

    with analysis3:
    
        st.metric(
            "Spectral Centroid",
            f"{spectral_centroid:.0f} Hz"
        )
    st.divider()
    
    # ---------------------------------------------------
    # EXTRACTED FEATURES
    # ---------------------------------------------------
    
    st.subheader(
        "🧠 Extracted Audio Features"
    )
    
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )
    
    mfcc_means = np.mean(
        mfccs,
        axis=1
    )
    
    feature_df = pd.DataFrame({
        "MFCC Feature": [
            f"MFCC_{i}"
            for i in range(1, 14)
        ],
        "Value": mfcc_means
    })
    
    st.dataframe(
        feature_df,
        use_container_width=True
    )
    
    fig_mfcc, ax_mfcc = plt.subplots(
        figsize=(10, 4)
    )
    
    ax_mfcc.bar(
        feature_df["MFCC Feature"],
        feature_df["Value"]
    )
    
    ax_mfcc.set_title(
        "MFCC Profile of Uploaded Audio"
    )
    
    plt.xticks(
        rotation=45
    )
    
    st.pyplot(fig_mfcc)

    if st.button("Predict Sound"):
        uploaded_file.seek(0)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "audio/wav"
            )
        }

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                # -------------------------
                # Save History
                # -------------------------

                st.session_state.history.append(
                    {
                        "File": uploaded_file.name,
                        "Prediction": CATEGORY_MAP.get(
                            result["prediction"],
                            result["prediction"]
                        ),
                        "Confidence": result["confidence"]
                    }
                )

                st.success(
                    "Prediction Complete"
                )

                st.subheader(
                    "🎯 Classification Result"
                )

                pred_col1, pred_col2 = st.columns(2)

                with pred_col1:

                    st.metric(
                        "Predicted Sound",
                        CATEGORY_MAP.get(
                            result["prediction"],
                            result["prediction"]
                        )
                    )

                with pred_col2:

                    st.metric(
                        "Confidence Score",
                        f"{result['confidence']} %"
                    )

                st.divider()

                st.subheader(
                    "🏆 Top Predictions"
                )
                
                top_predictions = result.get(
                    "top_predictions",
                    []
                )
                
                if len(top_predictions) > 0:
                
                    top_df = pd.DataFrame(
                        top_predictions
                    )
                
                    top_df.columns = [
                        "Sound Class",
                        "Probability (%)"
                    ]
                
                    st.dataframe(
                        top_df,
                        use_container_width=True
                    )
                
                    fig_top, ax_top = plt.subplots(
                        figsize=(8, 4)
                    )
                
                    ax_top.barh(
                        top_df["Sound Class"],
                        top_df["Probability (%)"]
                    )
                
                    ax_top.invert_yaxis()
                
                    ax_top.set_xlabel(
                        "Probability (%)"
                    )
                
                    ax_top.set_title(
                        "Top Model Predictions"
                    )
                
                    st.pyplot(
                        fig_top
                    )

                confidence = result["confidence"]

                st.progress(
                    confidence / 100
                )
                

                st.subheader(
                    "📈 Prediction Confidence Analysis"
                )
                
                if confidence >= 80:
                
                    st.success(
                        "Model is highly confident in this prediction."
                    )
                
                elif confidence >= 60:
                
                    st.warning(
                        "Model confidence is moderate. Similar classes may exist."
                    )
                
                else:
                
                    st.error(
                        "Low confidence prediction. Audio may contain overlapping sound patterns."
                    )
                st.subheader(
                    "🧠 Audio Interpretation"
                )
                
                if spectral_centroid > 3000:
                
                    brightness = "Bright / High Frequency"
                
                else:
                
                    brightness = "Low Frequency Dominant"
                
                st.info(
                    f"""
                Spectral Character: {brightness}
                
                Spectral Centroid: {spectral_centroid:.0f} Hz
                
                Zero Crossing Rate: {zero_crossing_rate:.4f}
                
                These characteristics help the Random Forest model distinguish environmental sounds.
                """
                )

                

                st.info(
                    f"""
Predicted Category:
{CATEGORY_MAP.get(result['prediction'], result['prediction'])}

Confidence:
{confidence:.2f}%

Model:
Random Forest Classifier
"""
                )

            else:

                st.error(
                    "Prediction Failed"
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

# ---------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------
st.divider()

st.subheader(
    "📈 Prediction Statistics"
)

stat1, stat2, stat3 = st.columns(3)

total_predictions = len(
    st.session_state.history
)

unique_classes = len(
    set(
        [
            row["Prediction"]
            for row in st.session_state.history
        ]
    )
) if total_predictions > 0 else 0

avg_confidence = round(
    sum(
        row["Confidence"]
        for row in st.session_state.history
    ) / total_predictions,
    2
) if total_predictions > 0 else 0

with stat1:

    st.metric(
        "Total Predictions",
        total_predictions
    )

with stat2:

    st.metric(
        "Unique Classes",
        unique_classes
    )

with stat3:

    st.metric(
        "Average Confidence",
        f"{avg_confidence}%"
    )
st.divider()

st.subheader(
    "📊 Prediction Distribution"
)

if len(st.session_state.history) > 0:

    prediction_counts = pd.DataFrame(
        st.session_state.history
    )["Prediction"].value_counts()

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        prediction_counts.index,
        prediction_counts.values
    )

    ax.set_ylabel(
        "Count"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    plt.xticks(
        rotation=45
    )

    st.pyplot(fig)

else:

    st.info(
        "No predictions available yet."
    )
st.subheader(
    "Prediction History"
)

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info(
        "No predictions made yet."
    )
st.divider()

st.markdown(
    """
<div style="text-align:center; padding:20px;">

### 🎵 Natural Sound Intelligence Platform

Built using FastAPI, Streamlit, Librosa and Scikit-Learn

Environmental Sound Classification Capstone Project

Version 1.0

</div>
""",
    unsafe_allow_html=True
)