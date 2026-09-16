# Adaptive Machine Learning-based Wireless Channel Intelligence

An adaptive machine learning framework for predictive wireless channel estimation, dynamic tracking, beam prediction, and link intelligence powered by DeepMIMO 3.5 GHz channel ray-tracing data.

🌐 **Live Interactive Web Demo**: [https://nidhishreddyy8-a11y.github.io/Adaptive-Machine-Learning-based-Wireless-Channel-Intelligence/](https://nidhishreddyy8-a11y.github.io/Adaptive-Machine-Learning-based-Wireless-Channel-Intelligence/)

---

## 📌 Overview

Wireless communications in 5G/6G systems require ultra-fast, dynamic adaptation to channel variations. This project demonstrates an end-to-end Machine Learning pipeline and an interactive dashboard for:
- **Line-of-Sight (LoS) Classification**: Classifies channel conditions (LoS vs. NLoS) using spatial coordinates and channel statistics.
- **Received Signal Strength (RSS) Prediction**: Regression models predicting signal power levels across mobile user locations.
- **Optimal Beam Prediction**: Identifies optimal beam indexes (64-beam codebook) for directional mmWave/sub-6GHz transmission.
- **Channel Compression Autoencoder**: Dimensionality reduction on high-dimensional CSI (Channel State Information) matrices.
- **Interactive Web Dashboard**: Real-time evaluation dashboard built with Flask and modern CSS/JS charts.

---

## 🚀 Key Features

- **Multi-Model Channel Prediction**:
  - LoS: Logistic Regression, Gradient Boosting Classifier, and Spatial KNN
  - RSS: Multi-Layer Perceptron (MLP), Gradient Boosting Regressor, and Spatial KNN
  - Beam: Multi-Layer Perceptron (MLP) and Random Forest
  - Channel Autoencoder: Neural network for CSI feedback compression
- **DeepMIMO Dataset Integration**: Validated on the `asu_campus_3p5` 3.5 GHz urban outdoor scenario.
- **Web UI & Visualization**: Full interactive interface for running single-point inference, trajectory tracking, and viewing model accuracy metrics.

---

## 📂 Project Structure

```text
├── app/
│   ├── app.py                     # Flask web server & inference API
│   ├── templates/
│   │   └── index.html             # Dashboard UI
│   └── static/
│       ├── css/style.css          # Frontend styling
│       ├── js/main.js             # Client-side charting and interactive logic
│       └── img/                   # Model performance & evaluation charts
├── models/                        # Trained model binaries and metadata
│   ├── ae.pkl                     # Autoencoder model
│   ├── beam_mlp.pkl               # Beam prediction MLP
│   ├── los_gbm.pkl / los_lr.pkl   # LoS classification models
│   ├── rss_gbm.pkl / rss_mlp.pkl  # RSS regression models
│   └── metadata.json              # Model metrics & feature schema
├── channel_intelligence_pipeline.ipynb  # End-to-end ML exploration notebook
├── train_pipeline.py              # Training script for all models
├── run_notebook.py                # Headless notebook execution script
├── requirements.txt               # Python package dependencies
├── start_server.bat               # Windows batch launcher for dashboard
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nidhishreddyy8-a11y/Adaptive-Machine-Learning-based-Wireless-Channel-Intelligence.git
cd Adaptive-Machine-Learning-based-Wireless-Channel-Intelligence
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option 1: Python CLI
```bash
python app/app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### Option 2: Windows Batch Launcher
Double-click or run:
```bat
start_server.bat
```

---

## 📊 Training the Models

To re-train the models using the dataset:
```bash
python train_pipeline.py
```
Or execute the Jupyter notebook:
```bash
jupyter notebook channel_intelligence_pipeline.ipynb
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
