# 🎵 Natural Sound Intelligence Platform

An end-to-end Machine Learning platform for Environmental Sound Classification using advanced audio signal processing, FastAPI, and Streamlit.

---

## 📌 Project Overview

The Natural Sound Intelligence Platform classifies environmental sounds from uploaded WAV audio files using a trained Random Forest model.

The platform combines:

* Audio Feature Extraction
* Machine Learning Classification
* Audio Visualization
* Dataset Analytics
* Model Explainability

into a single interactive dashboard.

---

## 🚀 Features

### 🎵 Audio Analysis

* WAV File Upload
* Audio Playback
* Duration Detection
* Sample Rate Detection
* Waveform Visualization
* Mel Spectrogram Generation

### 🔬 Signal Processing

* RMS Energy Analysis
* Zero Crossing Rate
* Spectral Centroid Analysis
* MFCC Feature Extraction

### 🤖 Machine Learning

* Random Forest Classifier
* Confidence Scoring
* Top-3 Prediction Ranking
* Prediction Explainability

### 📊 Analytics Dashboards

#### Model Analytics

* Feature Importance
* MFCC Analytics
* Chroma Analytics
* Correlation Heatmap
* Model Performance Comparison

#### Dataset Insights

* Category Distribution
* Dataset Statistics
* Feature Group Analysis
* Feature Distribution Analysis

---

## 🏗️ Project Architecture

```text
Audio Upload
      ↓
Feature Extraction
      ↓
Feature Scaling
      ↓
Random Forest Model
      ↓
Prediction Engine
      ↓
Explainability Layer
      ↓
Analytics Dashboard
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* Scikit-Learn
* Joblib

### Audio Processing

* Librosa
* NumPy

### Data Analysis

* Pandas
* Matplotlib
* Seaborn

---

## 📂 Dataset

Environmental Sound Dataset

Classes:

* Chirping Birds
* Thunderstorm
* Crow
* Pouring Water
* Water Drops
* Wind
* Frog
* Crackling Fire
* Rain
* Insects
* Rooster
* Sea Waves
* Crickets

Total Classes: 13

---

## 📈 Model Performance

| Metric                | Value         |
| --------------------- | ------------- |
| Model                 | Random Forest |
| Features              | 38            |
| Classes               | 13            |
| Test Accuracy         | 72.12%        |
| Full Dataset Accuracy | 94.42%        |

---

## ▶️ Installation

Clone repository:

```bash
git clone <repository-url>
cd Natural_Sound_Statistics_Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Backend

```bash
python -m uvicorn backend.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## ▶️ Run Frontend

```bash
streamlit run frontend/streamlit_app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## 📊 Dashboard Pages

### Home

* Audio Upload
* Prediction Engine
* Audio Analysis
* Prediction Explainability

### Model Analytics

* Feature Importance
* Correlation Analysis
* Feature Statistics

### Dataset Insights

* Dataset Overview
* Distribution Analysis
* Feature Analytics

### Project Report

* Technical Documentation
* Results Summary

---

## 🔮 Future Improvements

* Deep Learning Models (CNN / CRNN)
* Real-Time Audio Classification
* Audio Recording Support
* Cloud Deployment
* SHAP Explainability
* User Authentication

---

## 👨‍💻 Author

Ashish Kumar Patra

B.Tech Student

Natural Sound Intelligence Platform
