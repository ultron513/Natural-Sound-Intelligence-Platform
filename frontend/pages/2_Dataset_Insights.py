import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dataset Insights",
    page_icon="📂",
    layout="wide"
)

# ---------------------------------------------------
# LOAD REAL DATASET
# ---------------------------------------------------

data_df = pd.read_csv(
    "../outputs/advanced_audio_features.csv"
)

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title(
    "📂 Dataset Insights Dashboard"
)

st.markdown(
    """
Explore the structure and distribution of the environmental sound dataset.
"""
)

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

total_samples = len(data_df)

total_classes = data_df["category"].nunique()

total_features = len(data_df.columns) - 2

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Samples",
        total_samples
    )

with col2:
    st.metric(
        "Classes",
        total_classes
    )

with col3:
    st.metric(
        "Features",
        total_features
    )

st.divider()

# ---------------------------------------------------
# CATEGORY DISTRIBUTION
# ---------------------------------------------------

distribution_df = (
    data_df["category"]
    .value_counts()
    .reset_index()
)

distribution_df.columns = [
    "Category",
    "Samples"
]

st.subheader(
    "Distribution of Sound Categories"
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.bar(
    distribution_df["Category"],
    distribution_df["Samples"]
)

plt.xticks(
    rotation=60
)

ax.set_ylabel(
    "Number of Samples"
)

ax.set_xlabel(
    "Sound Category"
)

st.pyplot(fig)

st.divider()

# ---------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------

st.subheader(
    "Dataset Summary"
)

summary_df = pd.DataFrame({
    "Metric": [
        "Dataset Name",
        "Total Samples",
        "Classes",
        "Features",
        "Audio Format",
        "Model Used"
    ],
    "Value": [
        "Advanced Environmental Audio Dataset",
        str(total_samples),
        str(total_classes),
        str(total_features),
        "WAV",
        "Random Forest"
    ]
})

st.dataframe(
    summary_df,
    use_container_width=True
)

st.divider()

# ---------------------------------------------------
# SOUND CATEGORY EXPLORER
# ---------------------------------------------------

st.subheader(
    "Sound Category Explorer"
)

selected_category = st.selectbox(
    "Choose a Sound Category",
    sorted(
        data_df["category"].unique()
    )
)

category_info = {
    "chirping_birds": "Bird vocalization recordings collected from natural environments.",
    "thunderstorm": "Thunder and storm-related atmospheric sounds.",
    "crow": "Crow vocalization recordings.",
    "pouring_water": "Continuous flowing or pouring water sounds.",
    "water_drops": "Individual water drop recordings.",
    "wind": "Natural wind and air movement sounds.",
    "frog": "Frog vocalization recordings.",
    "crackling_fire": "Fire and burning wood sounds.",
    "rain": "Rainfall recordings.",
    "insects": "Various insect-generated sounds.",
    "rooster": "Rooster crowing recordings.",
    "sea_waves": "Ocean and sea wave sounds.",
    "crickets": "Cricket chirping recordings."
}

if selected_category in category_info:

    st.info(
        category_info[selected_category]
    )

else:

    st.info(
        f"{selected_category} recordings from the environmental audio dataset."
    )
st.divider()

# ---------------------------------------------------
# FEATURE GROUP ANALYTICS
# ---------------------------------------------------

st.subheader(
    "Feature Group Analytics"
)

mfcc_columns = [
    col
    for col in data_df.columns
    if col.startswith("mfcc_")
]

chroma_columns = [
    col
    for col in data_df.columns
    if col.startswith("chroma_")
]

contrast_columns = [
    col
    for col in data_df.columns
    if col.startswith("contrast_")
]

tonnetz_columns = [
    col
    for col in data_df.columns
    if col.startswith("tonnetz_")
]

feature_groups = pd.DataFrame({
    "Feature Group": [
        "MFCC",
        "Chroma",
        "Contrast",
        "Tonnetz"
    ],
    "Count": [
        len(mfcc_columns),
        len(chroma_columns),
        len(contrast_columns),
        len(tonnetz_columns)
    ]
})

fig2, ax2 = plt.subplots(
    figsize=(8, 4)
)

ax2.bar(
    feature_groups["Feature Group"],
    feature_groups["Count"]
)

ax2.set_ylabel(
    "Number of Features"
)

ax2.set_title(
    "Feature Engineering Breakdown"
)

st.pyplot(fig2)
st.divider()

# ---------------------------------------------------
# DATASET FEATURE STATISTICS
# ---------------------------------------------------

st.subheader(
    "Dataset Feature Statistics"
)

feature_columns = [
    col
    for col in data_df.columns
    if col not in [
        "filename",
        "category"
    ]
]

mean_value = round(
    data_df[feature_columns]
    .mean()
    .mean(),
    4
)

std_value = round(
    data_df[feature_columns]
    .std()
    .mean(),
    4
)

max_value = round(
    data_df[feature_columns]
    .max()
    .max(),
    4
)

min_value = round(
    data_df[feature_columns]
    .min()
    .min(),
    4
)

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:
    st.metric(
        "Average Feature Value",
        mean_value
    )

with stat2:
    st.metric(
        "Average Std Dev",
        std_value
    )

with stat3:
    st.metric(
        "Maximum Value",
        max_value
    )

with stat4:
    st.metric(
        "Minimum Value",
        min_value
    )
st.divider()

# ---------------------------------------------------
# FEATURE DISTRIBUTIONS
# ---------------------------------------------------

st.subheader(
    "Feature Distribution Analysis"
)

feature_type = st.selectbox(
    "Select Feature Family",
    [
        "MFCC",
        "Chroma",
        "Contrast",
        "Tonnetz"
    ]
)

if feature_type == "MFCC":

    selected_columns = mfcc_columns

elif feature_type == "Chroma":

    selected_columns = chroma_columns

elif feature_type == "Contrast":

    selected_columns = contrast_columns

else:

    selected_columns = tonnetz_columns

distribution_values = (
    data_df[selected_columns]
    .mean()
)

fig3, ax3 = plt.subplots(
    figsize=(10, 4)
)

ax3.bar(
    distribution_values.index,
    distribution_values.values
)

ax3.set_title(
    f"{feature_type} Feature Distribution"
)

ax3.set_ylabel(
    "Average Value"
)

plt.xticks(
    rotation=45
)

st.pyplot(fig3)