# Air Traffic Congestion Prediction System

A spatio-temporal deep learning pipeline for forecasting U.S. airspace congestion using real-world ADS-B flight trajectory data.

## Overview

This project predicts short-term air traffic congestion by converting raw ADS-B flight state vectors into 4D congestion tensors (latitude × longitude × altitude × time) and training spatio-temporal deep learning models to forecast congestion intensity across airport and regional levels. The system also includes explainable AI visualizations to interpret model predictions.

Developed as a team project (team of 4), guided by Dr. KVSN. Rama Rao.

## Features

- **Spatio-temporal forecasting** on real-world ADS-S/ADS-B trajectory data (2019–2021)
- **4D congestion tensor construction** from raw flight state vectors
- **Multi-model comparison**: ConvLSTM, 3D CNN, and TSTD-GCN architectures for congestion intensity forecasting
- **Explainable AI visualizations**: satellite map overlays, 3D airspace views, and saliency maps to interpret predictions
- **Interactive dashboard** deployed on Streamlit Community Cloud for exploring predictions and flight animations

## Dataset

- **Source**: OpenSky Network ADS-B data
- **Scale**: 50,000+ U.S. aircraft trajectories
- **Time range**: 2019–2021
- Supplementary real-time data integrations: OpenWeatherMap, AviationStack (for the companion live-dashboard build)

> _Note: Add exact dataset file paths / download instructions here once finalized._

## Model Architecture

| Model | Purpose | Notes |
|---|---|---|
| ConvLSTM | Congestion intensity forecasting | Achieved lower MSE and superior temporal consistency vs. 3D CNN |
| 3D CNN | Congestion intensity forecasting | Baseline spatio-temporal comparison model |
| TSTD-GCN | Spatio-temporal graph-based forecasting | Core architecture for the formal project report |

> _Note: Insert exact MSE / evaluation metric values here once you have them documented from your project report._

## Tech Stack

- **Language**: Python
- **Deep Learning**: PyTorch
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly, saliency overlays, satellite map rendering
- **Dashboard/Deployment**: Streamlit (Streamlit Community Cloud)
- **APIs**: OpenSky Network, OpenWeatherMap, AviationStack

## Project Structure

```
air-traffic-congestion-prediction/
├── data/                  # Raw and processed ADS-B trajectory data
├── models/                # ConvLSTM, 3D CNN, TSTD-GCN model definitions
├── notebooks/             # Exploratory analysis and training notebooks
├── visualizations/        # Saliency overlays, satellite/3D airspace views
├── app.py                 # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

> _Note: Update this to match your actual repo structure before publishing._

## Installation

```bash
git clone https://github.com/Bishwanath20/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

> _Note: Add any data preprocessing / model training commands here (e.g., `python train.py --model convlstm`)._

## Results

- ConvLSTM outperformed the 3D CNN baseline in both MSE and temporal consistency of congestion forecasts.
- Explainable AI visualizations (saliency overlays, 3D airspace views) provided interpretable insight into which spatio-temporal regions most influenced model predictions.

> _Note: Add specific quantitative results/screenshots once available._

## Contributors

- Bishwanath Patra
- Vaddadi Adithya

**Guided by**: Dr. KVSN. Rama Rao

## License

> _Add license type here (e.g., MIT) if applicable._
