import numpy as np
import pandas as pd
import tensorflow as tf
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sklearn.preprocessing import MinMaxScaler
import sys
from datetime import datetime

# Configuration
MODEL_PATH = "stock_predictor_fixed.h5"
DATA_PATH = "final_dataset_ALL.csv"
TICKERS = ['AAPL', 'GOOGL', 'MSFT']
LOOKBACK = 60

# Elasticsearch configuration
ES_HOST = "localhost"  # or your Elasticsearch server address
ES_PORT = 9200
ES_INDEX = "stock_predictions"

class CustomAttention(tf.keras.layers.Attention):
    def __init__(self, **kwargs):
        kwargs.pop('causal', None)
        super().__init__(**kwargs)

def load_custom_model(path):
    """Handle both LSTM and Attention layer issues"""
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'LSTM': lambda *args, **kwargs: tf.keras.layers.LSTM(*args, **{k: v for k, v in kwargs.items() if k != 'time_major'}),
            'Attention': CustomAttention
        }
    )

def connect_elasticsearch():
    """Create and return Elasticsearch connection"""
    
    es = Elasticsearch(
    "http://localhost:9200",
    verify_certs=False)  # dev only
    # Test connection
    if not es.ping():
        raise ValueError("Connection to Elasticsearch failed")
    return es

def create_index(es):
    """Create Elasticsearch index if it doesn't exist"""
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
    """Prepare prediction data for Elasticsearch"""
    return {
        "ticker": ticker,
        "prediction": action,
        "confidence": float(confidence),
        "last_date": last_date,
        "timestamp": datetime.utcnow(),
        "model_version": "1.0"  # You can version your model
    }

def load_data(ticker):
    """Load and preprocess stock data from CSV"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Convert column names to lowercase for case-insensitive matching
        df.columns = df.columns.str.lower()
        
        # Filter data for the specific ticker
        df = df[df['ticker'] == ticker.upper()].sort_values('date')
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Select relevant columns - now including mood_score as the 6th feature
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'mood_score']
        if not all(col in df.columns for col in feature_cols):
            raise ValueError("Missing required columns in the data")
        
        # Fill any missing mood_score values with 0
        df['mood_score'] = df['mood_score'].fillna(0)
        
        # Normalize data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[feature_cols].values)
        
        # Create sequences for LSTM - now using 7 timesteps instead of 60
        X = []
        for i in range(7, len(data)):
            X.append(data[i-7:i])
        X = np.array(X)
        
        return X, df['date'].values[-1], scaler
        
    except Exception as e:
        print(f"⚠️ Error loading data for {ticker}: {str(e)}")
        return None, None, None

def predict_stock(model, X):
    """Make prediction using the LSTM model"""
    try:
        # Reshape input to match model's expected shape (batch_size, 7, 6)
        prediction = model.predict(X[-1:].reshape(1, 7, 6), verbose=0)[0][0]
        return "BUY" if prediction > 0.5 else "SELL", abs(prediction - 0.5) * 2
    except Exception as e:
        print(f"⚠️ Prediction error: {str(e)}")
        return None, None

def run_predictions():
    """Main prediction workflow"""
    print("\n🚀 Stock Prediction System")
    print("="*50)
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    
    try:
        # Connect to Elasticsearch
        es = connect_elasticsearch()
        create_index(es)
        print(f"✅ Connected to Elasticsearch at {ES_HOST}:{ES_PORT}")
        
        # Load model
        model = load_custom_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        predictions = []
        
        print("\n🔮 Stock Predictions:")
        print("="*40)
        print(f"{'Ticker':<6} | {'Prediction':<10} | {'Confidence':<10} | Last Date")
        print("-"*40)
        
        for ticker in TICKERS:
            X, last_date, scaler = load_data(ticker)
            if X is not None:
                action, confidence = predict_stock(model, X)
                if action:
                    print(f"{ticker:<6} | {action:<10} | {confidence:.1%}       | {last_date}")
                    # Prepare document for Elasticsearch
                    doc = prepare_document(ticker, action, confidence, last_date)
                    predictions.append(doc)
        
        # Bulk insert to Elasticsearch
        if predictions:
            actions = [
                {
                    "_index": ES_INDEX,
                    "_source": doc
                }
                for doc in predictions
            ]
            success, _ = bulk(es, actions)
            print(f"\n✅ Successfully loaded {success} predictions to Elasticsearch")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    run_predictions()