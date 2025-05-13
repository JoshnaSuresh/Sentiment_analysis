## Sentiment-Driven Stock Market Analysis and Prediction Using Elasticsearch and Natural Language Processing
This project predicts short-term stock movements by combining **technical indicators** with **news-based sentiment analysis**. The system uses deep learning (LSTM + Attention) and integrates with **Elasticsearch** for real-time prediction indexing and visualization.

## Overview

Traditional stock prediction models rely on historical price data alone. This system goes further by incorporating sentiment data extracted from news headlines, offering a richer and more dynamic view of the market. The model supports predictions for stocks like **AAPL**, **GOOGL**, and **MSFT**.

Key components include:

1. Sentiment extraction using `TextBlob`
2. Technical indicator computation (RSI, MACD, SMA)
3. LSTM + Attention model
4. Bulk prediction indexing into Elasticsearch
5. Real-time search and Kibana visualization

## Tech Stack

  **Python 3.10**
  **TensorFlow / Keras**
  **Pandas / NumPy / scikit-learn**
  **TextBlob**
  **pandas_ta** (technical indicators)
  **yfinance** (stock data)
  **Elasticsearch (v8+)**
  **Docker** (for Elasticsearch setup)
  **Kibana** (for dashboards)

## Setup Instructions

## 1. Clone the Repository
git clone https://github.com/JoshnaSuresh/Sentiment_analysis.git
cd stock-sentiment-predictor

## 2. Setup Python Environment
python -m venv venv
source venv/bin/activate  # or use `venv\\Scripts\\activate` on Windows
pip install -r requirements.txt

## 3. Start Elasticsearch (via Docker)
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.9.0

## 4. Run Script
python predict.py
