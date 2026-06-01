import streamlit as st

st.set_page_config(
    page_title="Project Report",
    page_icon="📄",
    layout="wide"
)

st.title(
    "📄 Natural Sound Intelligence Platform"
)

st.markdown(
    """
### End-to-End Environmental Sound Classification System

This project demonstrates how machine learning can be used to classify
natural environmental sounds using advanced audio feature engineering,
Random Forest classification, FastAPI backend services, and a
Streamlit-based analytics dashboard.
"""
)

st.divider()

# ---------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------

st.header("🎯 Project Overview")

st.write(
    """
The objective of this project is to automatically identify
environmental sounds from uploaded audio recordings.

Users can upload a WAV file and receive a predicted sound category
along with model confidence.
"""
)

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

st.header("📂 Dataset")

st.write(
    """
Dataset Used: ESC-50 Environmental Sound Dataset

Filtered Categories:
- Birds
- Rain
- Water
- Fire
- Insects
- Wind
- Sea Waves
- Thunderstorm
and other environmental sounds.

Total Samples: 520
"""
)

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

st.header("🎧 Feature Engineering")

st.write(
    """
Audio features were extracted using Librosa.

Extracted Features:
- MFCC Features
- Chroma Features
- Spectral Contrast
- Tonnetz Features

Total Features:
38
"""
)

# ---------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------

st.header("🤖 Model Comparison")

comparison = {
    "Model": [
        "Random Forest",
        "SVM",
        "Gradient Boosting",
        "KNN"
    ],
    "Accuracy": [
        "72.12%",
        "66.35%",
        "60.58%",
        "56.73%"
    ]
}

st.table(comparison)
st.divider()

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------

st.header("📈 Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Test Accuracy",
        "72.12%"
    )

with col2:
    st.metric(
        "Full Dataset Accuracy",
        "94.42%"
    )

st.success(
    """
Random Forest achieved the highest classification performance
among all evaluated machine learning models and was selected
for deployment.
"""
)

# ---------------------------------------------------
# SYSTEM ARCHITECTURE
# ---------------------------------------------------

st.header("🏗️ System Architecture")

st.code(
    """
User Uploads Audio
        ↓
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
Feature Extraction
        ↓
Random Forest Model
        ↓
Prediction Result
        ↓
Interactive Dashboard
"""
)

# ---------------------------------------------------
# TECHNOLOGY STACK
# ---------------------------------------------------

st.header("🛠️ Technology Stack")

stack_col1, stack_col2 = st.columns(2)

with stack_col1:

    st.write(
        """
Frontend:
- Streamlit

Backend:
- FastAPI

Machine Learning:
- Scikit-Learn
"""
    )

with stack_col2:

    st.write(
        """
Audio Processing:
- Librosa

Data Analysis:
- Pandas
- NumPy

Visualization:
- Matplotlib
"""
    )

# ---------------------------------------------------
# FUTURE SCOPE
# ---------------------------------------------------

st.header("🚀 Future Scope")

st.write(
    """
Future enhancements may include:

- Deep Learning based classification
- Real-time microphone input
- Mobile application deployment
- Cloud deployment
- Live audio stream analysis
- Larger environmental sound datasets
- Explainable AI integration
"""
)

# ---------------------------------------------------
# PROJECT SUMMARY
# ---------------------------------------------------

st.divider()

st.success(
    """
Natural Sound Intelligence Platform successfully demonstrates
an end-to-end machine learning workflow including:

✓ Audio Processing

✓ Feature Engineering

✓ Model Training

✓ API Development

✓ Interactive Dashboard Design

✓ Real-Time Sound Classification
"""
)