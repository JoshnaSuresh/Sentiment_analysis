import numpy as np
import tensorflow as tf
from elasticsearch import Elasticsearch
from datetime import datetime, timezone
import urllib3
import warnings

# Configuration
ES_CONFIG = {
    "hosts": ["http://localhost:9200"],
    "request_timeout": 30
}

# Suppress warnings
urllib3.disable_warnings()
warnings.filterwarnings("ignore", module="elasticsearch")

def init_elasticsearch():
    """Initialize and verify Elasticsearch connection"""
    try:
        es = Elasticsearch(**ES_CONFIG)
        if es.ping():
            print("✅ Connected to Elasticsearch")
            print(f"Cluster: {es.info()['cluster_name']} (v{es.info()['version']['number']})")
            return es
        raise ConnectionError("Failed to ping Elasticsearch")
    except Exception as e:
        print(f"❌ Elasticsearch Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify Docker is running: 'docker ps'")
        print("2. Check Elasticsearch logs: 'docker logs elasticsearch'")
        print("3. Test connection manually: 'curl http://localhost:9200'")
        return None

def save_prediction(es, ticker, direction, confidence):
    """Save prediction to Elasticsearch"""
    if not es:
        return

    doc = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "prediction": direction.lower(),
        "confidence": float(confidence.strip('%'))/100,
        "model_version": "lstm_attention_v1",
        "next_trading_day": datetime.now(timezone.utc).strftime("%Y-%m-%d")
    }

    try:
        res = es.index(
            index="stock_predictions",
            document=doc,
            refresh=True
        )
        if res['result'] == 'created':
            print(f"   ↳ Saved {ticker} prediction (ID: {res['_id']})")
        else:
            print(f"⚠️ Failed to index {ticker} prediction")
    except Exception as e:
        print(f"⚠️ Elasticsearch write error: {str(e)}")

def run_predictions():
    """Main prediction workflow"""
    es = init_elasticsearch()
    
    try:
        model = tf.keras.models.load_model("stock_predictor_fixed.h5")
        data = np.load("training_data.npz")
        X, y = data['X'], data['y']
        
        if len(X.shape) == 4:
            X = X.reshape(-1, 7, 6)
        print(f"\n📊 Data loaded - X: {X.shape}, y: {y.shape}")
    except Exception as e:
        print(f"❌ Model/Data Error: {e}")
        return

    tickers = ['AAPL', 'GOOGL', 'MSFT']
    print(f"\n🔮 Next Trading Day Predictions")
    print("="*55)
    print(f"| {'Ticker':<6} | {'Prediction':<10} | {'Confidence':<10} | Trend  |")
    print("|-------|------------|------------|-------|")

    for i, ticker in enumerate(tickers):
        try:
            stock_data = X[i*len(X)//3 : (i+1)*len(X)//3][-1:]
            pred = model.predict(stock_data, verbose=0)
            prob = pred[0][0]
            direction = "↑ UP" if prob > 0.5 else "↓ DOWN"
            trend = ("🚀" if prob > 0.7 else
                    "↗" if prob > 0.6 else
                    "💥" if prob < 0.3 else
                    "↘" if prob < 0.4 else "➡")
            
            print(f"| {ticker:<6} | {direction:<10} | {prob:.1%}    | {trend:<5} |")
            save_prediction(es, ticker, direction, f"{prob:.1%}")
            
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {str(e)}")

if __name__ == "__main__":
    run_predictions()