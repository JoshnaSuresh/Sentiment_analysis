import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sklearn.preprocessing import MinMaxScaler
import sys
from datetime import datetime, timedelta
from textblob import TextBlob
import pandas_ta as ta
import time
from yahoo_fin import stock_info
from datetime import datetime, timedelta

# Configuration
MODEL_PATH = "stock_predictor_10feature.h5"
TICKERS = ['AAPL', 'GOOGL', 'MSFT']
LOOKBACK = 60
HISTORY_DAYS = 180

# Elasticsearch configuration
ES_HOST = "localhost"
ES_PORT = 9200
ES_INDEX = "stock_predictions"

class CustomAttention(tf.keras.layers.Attention):
    def __init__(self, **kwargs):
        kwargs.pop('causal', None)
        super().__init__(**kwargs)

def load_custom_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'LSTM': lambda *args, **kwargs: tf.keras.layers.LSTM(*args, **{k: v for k, v in kwargs.items() if k != 'time_major'}),
            'Attention': CustomAttention
        }
    )

def connect_elasticsearch():
    es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}", verify_certs=False)
    if not es.ping():
        raise ValueError("Connection to Elasticsearch failed")
    return es

def create_index(es):
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(
            index=ES_INDEX,
            body={
                "mappings": {
                    "properties": {
                        "ticker": {"type": "keyword"},
                        "prediction": {"type": "keyword"},
                        "confidence": {"type": "float"},
                        "last_date": {"type": "date"},
                        "timestamp": {"type": "date"},
                        "model_version": {"type": "keyword"}
                    }
                }
            }
        )

def prepare_document(ticker, action, confidence, last_date):
    return {
        "ticker": ticker,
        "prediction": action,
        "confidence": float(confidence),
        "last_date": pd.to_datetime(last_date).to_pydatetime().isoformat(),
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": "1.0"
    }

def fetch_sentiment_score(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return 0
        headlines = " ".join([item['title'] for item in news[:5]])
        analysis = TextBlob(headlines)
        return analysis.sentiment.polarity
    except:
        return 0

def fetch_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    try:
        time.sleep(1)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        df['ticker'] = ticker
        df['mood_score'] = fetch_sentiment_score(ticker)

        # Add technical indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_50'] = ta.sma(df['close'], length=50)

        # Add trade signal labels: 1 if price increases 2% in next 5 days, else 0
        df['target'] = (df['close'].shift(-5) > df['close'] * 1.02).astype(int)

        df = df.dropna()

        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'mood_score', 'rsi', 'macd', 'sma_10', 'sma_50', 'target', 'ticker']]
        return df
    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {str(e)}")
        return None

def fetch_stock_data1(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    attempt = 0
    max_attempts = 5  # Maximum number of retry attempts
    while attempt < max_attempts:
        try:
            print(f"Fetching data for {ticker} (Attempt {attempt + 1})...")
            time.sleep(3)  # Base delay
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            df['ticker'] = ticker
            df['mood_score'] = fetch_sentiment_score(ticker)

            # Add technical indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_50'] = ta.sma(df['close'], length=50)

            # Add trade signal labels: 1 if price increases 2% in next 5 days, else 0
            df['target'] = (df['close'].shift(-5) > df['close'] * 1.02).astype(int)

            df = df.dropna()

            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'mood_score', 'rsi', 'macd', 'sma_10', 'sma_50', 'target', 'ticker']]
            return df
        except Exception as e:
            print(f"⚠️ Error fetching data for {ticker}: {str(e)}")
            attempt += 1
            if attempt < max_attempts:
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, limit to 30 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)  # Wait before retrying
            else:
                print(f"Failed to fetch data for {ticker} after {max_attempts} attempts.")
                return None

def prepare_training_data(df):
    try:
        if df is None or df.empty:
            raise ValueError("No data provided")
        if len(df) < LOOKBACK:
            raise ValueError(f"Not enough data points (need at least {LOOKBACK}, got {len(df)})")
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'mood_score', 'rsi', 'macd', 'sma_10', 'sma_50']
        data = df[feature_cols].values
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        X = [data[i-7:i] for i in range(7, len(data))]
        return np.array(X), df['date'].values[-1], scaler
    except Exception as e:
        print(f"⚠️ Data preparation error: {str(e)}")
        return None, None, None

def predict_stock(model, X):
    try:
        if X is None or len(X) == 0:
            raise ValueError("No input data provided")
        prediction = model.predict(X[-1:].reshape(1, 7, 10), verbose=0)[0][0]
        adjusted = np.clip((prediction - 0.5) * 1.5 + 0.5, 0, 1)
        return "BUY" if adjusted > 0.5 else "SELL", abs(adjusted - 0.5) * 2
    except Exception as e:
        print(f"⚠️ Prediction error: {str(e)}")
        return None, None

def run_predictions():
    print("\n🚀 Stock Prediction System")
    print("="*50)
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    try:
        es = connect_elasticsearch()
        create_index(es)
        print(f"✅ Connected to Elasticsearch at {ES_HOST}:{ES_PORT}")
        model = load_custom_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        predictions = []
        print("\n🔮 Stock Predictions:")
        print("="*40)
        print(f"{'Ticker':<6} | {'Prediction':<10} | {'Confidence':<10} | Last Date")
        print("-"*40)
        for ticker in TICKERS:
            df = fetch_stock_data(ticker)
            X, last_date, scaler = prepare_training_data(df)
            if X is not None:
                action, confidence = predict_stock(model, X)
                if action:
                    print(f"{ticker:<6} | {action:<10} | {confidence:.1%}       | {last_date}")
                    doc = prepare_document(ticker, action, confidence, last_date)
                    predictions.append(doc)
        if predictions:
            actions = [{"_index": ES_INDEX, "_source": doc} for doc in predictions]
            success, _ = bulk(es, actions)
            print(f"✅ Saved {len(predictions)} predictions to Elasticsearch index '{ES_INDEX}'")
            print(f"\n✅ Successfully loaded {success} predictions to Elasticsearch")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    run_predictions()
