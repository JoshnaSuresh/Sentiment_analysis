import numpy as np
import pandas as pd
import tensorflow as tf
from alpha_vantage.timeseries import TimeSeries
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from textblob import TextBlob
import pandas_ta as ta
import time
import warnings

warnings.filterwarnings("ignore")

# Configuration
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"
MODEL_PATH = "stock_predictor_10feature.h5"
TICKERS = ['AAPL', 'GOOGL', 'MSFT']
LOOKBACK = 60
HISTORY_DAYS = 180

es = Elasticsearch("http://localhost:9200")
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

def fetch_sentiment_score(ticker):
    try:
        text = f"{ticker} stock news"
        sentiment = TextBlob(text).sentiment.polarity
        return sentiment
    except Exception as e:
        print(f"⚠️ Error fetching sentiment score for {ticker}: {e}")
        return 0

def fetch_stock_data(ticker):
    print(f"Fetching stock data for {ticker}...")
    try:
        print(f"Requesting daily stock data for {ticker}...")
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        df = data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '6. volume': 'volume'
        })[['open', 'high', 'low', 'close', 'volume']]

        df = df.sort_index()  # Ensure data is sorted by date ascending
        df = df.last(f"{HISTORY_DAYS}D")
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        df.rename(columns={'date': 'datetime'}, inplace=True)
        df['ticker'] = ticker
        df['mood_score'] = fetch_sentiment_score(ticker)
        return df
    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {e}")
        return None

def prepare_features(df):
    try:
        df['sma'] = ta.sma(df['close'], length=10)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df = df.dropna()
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma', 'rsi', 'macd', 'mood_score']
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df[feature_cols].values[-LOOKBACK:]
    except Exception as e:
        print(f"⚠️ Data preparation error: {e}")
        return None

def predict(df_sequence):
    if df_sequence is None or df_sequence.shape[0] < LOOKBACK:
        return None
    try:
        input_data = np.expand_dims(df_sequence, axis=0)
        prediction = model.predict(input_data)[0][0]
        return prediction
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None

def save_to_elasticsearch(ticker, prediction):
    doc = {
        'ticker': ticker,
        'prediction': float(prediction),
        'timestamp': datetime.utcnow()
    }
    try:
        es.index(index="stock_predictions", document=doc)
        print(f"✅ Saved prediction for {ticker} to Elasticsearch.")
    except Exception as e:
        print(f"⚠️ Elasticsearch error: {e}")

# Main execution
for ticker in TICKERS:
    df = fetch_stock_data(ticker)
    if df is None or df.empty:
        print(f"No data available for {ticker}")
        continue
    features = prepare_features(df)
    pred = predict(features)
    if pred is not None:
        print(f"🔮 Prediction for {ticker}: {pred:.4f}")
        save_to_elasticsearch(ticker, pred)
    else:
        print(f"⚠️ Prediction skipped for {ticker}")
