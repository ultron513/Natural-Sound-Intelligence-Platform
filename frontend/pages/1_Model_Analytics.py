import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
audio_df = pd.read_csv(
   OUTPUTS_DIR / "advanced_audio_features.csv"
)
st.set_page_config(
    page_title="Model Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Analytics Dashboard")

st.markdown(
    """
Explore how the Random Forest model makes predictions
and which audio features contribute most to classification.
"""
)

# ---------------------------------------------------
# REAL FEATURE IMPORTANCE
# ---------------------------------------------------

st.subheader(
    "Top Audio Features (Real Model Importance)"
)

model = joblib.load(
    OUTPUTS_DIR / "rf_advanced_model.pkl"
)

feature_names = [
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4',
    'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8',
    'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12',
    'mfcc_13',

    'chroma_1', 'chroma_2', 'chroma_3',
    'chroma_4', 'chroma_5', 'chroma_6',
    'chroma_7', 'chroma_8', 'chroma_9',
    'chroma_10', 'chroma_11', 'chroma_12',

    'contrast_1', 'contrast_2', 'contrast_3',
    'contrast_4', 'contrast_5', 'contrast_6',
    'contrast_7',

    'tonnetz_1', 'tonnetz_2', 'tonnetz_3',
    'tonnetz_4', 'tonnetz_5', 'tonnetz_6'
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(10)
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Importance"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Top 10 Most Important Features"
)

st.pyplot(fig)
st.divider()

st.subheader(
    "Model Performance Comparison"
)

performance_df = pd.read_csv(
    "../outputs/model_comparison.csv"
)

fig2, ax2 = plt.subplots(
    figsize=(8, 4)
)

ax2.bar(
    performance_df["Model"],
    performance_df["Accuracy"]
)

ax2.set_ylabel(
    "Accuracy"
)

ax2.set_ylim(
    0.5,
    0.8
)

st.pyplot(fig2)
st.divider()

st.subheader(
    "Average MFCC Profile (Real Dataset)"
)

mfcc_columns = [
    f"mfcc_{i}"
    for i in range(1, 14)
]

mfcc_means = (
    audio_df[mfcc_columns]
    .mean()
)

fig3, ax3 = plt.subplots(
    figsize=(10, 4)
)

ax3.plot(
    mfcc_columns,
    mfcc_means.values,
    marker="o"
)

ax3.set_ylabel(
    "Average Value"
)

ax3.set_title(
    "Average MFCC Values Across Dataset"
)

plt.xticks(
    rotation=45
)

st.pyplot(fig3)
st.divider()

st.subheader(
    "Average Chroma Features (Real Dataset)"
)

chroma_columns = [
    f"chroma_{i}"
    for i in range(1, 13)
]

chroma_means = (
    audio_df[chroma_columns]
    .mean()
)

fig4, ax4 = plt.subplots(
    figsize=(10, 4)
)

ax4.bar(
    chroma_columns,
    chroma_means.values
)

ax4.set_ylabel(
    "Average Value"
)

ax4.set_title(
    "Average Chroma Values Across Dataset"
)

plt.xticks(
    rotation=45
)

st.pyplot(fig4)
st.divider()

st.subheader(
    "Feature Correlation Heatmap (Real Dataset)"
)

feature_columns = [
    col
    for col in audio_df.columns
    if col not in [
        "filename",
        "category"
    ]
]

corr_df = (
    audio_df[feature_columns]
    .corr()
)

fig5, ax5 = plt.subplots(
    figsize=(12, 10)
)

sns.heatmap(
    corr_df,
    cmap="coolwarm",
    center=0,
    ax=ax5
)

ax5.set_title(
    "Feature Correlation Matrix"
)

st.pyplot(fig5)