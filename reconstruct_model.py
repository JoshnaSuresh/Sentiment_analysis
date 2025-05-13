import numpy as np
import tensorflow as tf
from elasticsearch import Elasticsearch
from datetime import datetime, timezone, timedelta
import urllib3
import warnings
import sys
import time
import os
from tensorflow.keras.layers import LSTM, Dense, Input, Layer
from tensorflow.keras.models import Model, load_model
import h5py

# Configuration
ES_CONFIG = {
    "hosts": ["http://localhost:9200"],
    "timeout": 30,
    "headers": {"Accept": "application/json", "Content-Type": "application/json"}
}

# Suppress warnings
urllib3.disable_warnings()
warnings.filterwarnings("ignore", module="elasticsearch")

class CompatibleAttention(Layer):
    """Custom attention layer for version compatibility"""
    def __init__(self, **kwargs):
        kwargs.pop('dropout', None)
        kwargs.pop('use_scale', None)
        kwargs.pop('score_mode', None)
        kwargs.pop('causal', None)
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            name='kernel'
        )
        super().build(input_shape)
        
    def call(self, inputs):
        transformed = tf.matmul(inputs, self.kernel)
        attention = tf.nn.softmax(transformed, axis=1)
        return tf.reduce_sum(inputs * attention, axis=1)

def init_elasticsearch():
    """Initialize and verify Elasticsearch connection"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            es = Elasticsearch(**ES_CONFIG)
            if es.ping():
                print("✅ Connected to Elasticsearch")
                print(f"Cluster: {es.info()['cluster_name']} (v{es.info()['version']['number']})")
                return es
            raise ConnectionError("Failed to ping Elasticsearch")
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Elasticsearch Error: {e}")
                print("\nTroubleshooting:")
                print("1. Verify Docker is running: 'docker ps'")
                print("2. Check Elasticsearch logs: 'docker logs es01'")
                print("3. Test connection manually: 'curl http://localhost:9200'")
                return None
            print(f"⚠️ Connection failed (Attempt {attempt + 1}/{max_retries}), retrying...")
            time.sleep(2 * (attempt + 1))

def reconstruct_model(model_path, input_shape=(7, 6)):
    """Rebuild model architecture with compatible layers"""
    print("\n🔨 Reconstructing model architecture...")
    
    # Build new model
    inputs = Input(shape=input_shape)
    x = LSTM(256, return_sequences=True)(inputs)  # From your model inspection
    x = CompatibleAttention()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    # Try loading weights
    try:
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        print("✅ Model reconstructed with partial weights")
    except Exception as e:
        print(f"⚠️ Partial weight loading failed: {str(e)}")
        print("Proceeding with initialized weights")
    
    return model

def load_model_safely(model_path):
    """Attempt multiple loading strategies"""
    try:
        # Try with custom attention layer
        model = load_model(
            model_path,
            custom_objects={'Attention': CompatibleAttention},
            compile=False
        )
        print("✅ Model loaded with compatibility layer")
        return model
    except Exception as e:
        print(f"⚠️ Compatible load failed: {str(e)}")
        return reconstruct_model(model_path)

def run_predictions():
    """Main prediction workflow"""
    print("\n🚀 Starting Stock Prediction System")
    print("="*50)
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    
    # Initialize Elasticsearch
    es = init_elasticsearch()
    
    try:
        # Load model
        model = load_model_safely("stock_predictor_fixed.h5")
        model.summary()
        
        # Load data
        data = np.load("training_data.npz")
        X = data['X'].reshape(-1, 7, 6)  # Adjust shape as needed
        print(f"\n📊 Data shape: {X.shape}")
        
        # Make predictions
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        batch_size = len(X) // len(tickers)
        
        print("\n🔮 Predictions:")
        print("="*40)
        for i, ticker in enumerate(tickers):
            batch = X[i*batch_size : (i+1)*batch_size]
            pred = model.predict(batch[-1:], verbose=0)[0][0]
            direction = "↑ BUY" if pred > 0.5 else "↓ SELL"
            confidence = abs(pred - 0.5) * 2  # Convert to 0-1 range
            
            print(f"{ticker}: {direction} (Confidence: {confidence:.1%})")
            if es:
                save_prediction(es, ticker, direction, f"{confidence:.1%}")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
def save_prediction(es, ticker, direction, confidence):
    """Save prediction with enhanced error handling"""
    if not es:
        return False

    doc = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "prediction": direction.lower(),
        "confidence": float(confidence.strip('%'))/100,
        "model_version": "lstm_attention_v1",
        "next_trading_day": (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    }

    try:
        res = es.index(
            index="stock_predictions",
            body=doc,
            refresh=True
        )
        if res.get('result') in ['created', 'updated']:
            print(f"   ↳ Saved prediction for {ticker} (ID: {res['_id']})")
            return True
        print(f"⚠️ Unexpected response: {res}")
    except Exception as e:
        print(f"⚠️ Failed to save prediction: {str(e)}")
        if "index_not_found_exception" in str(e):
            print("   Try creating the index first or check permissions")
    return False
if __name__ == "__main__":
    run_predictions()