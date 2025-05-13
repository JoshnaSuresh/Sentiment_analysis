import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import pandas_ta as ta
from textblob import TextBlob

TICKERS = ['AAPL', 'GOOGL', 'MSFT']
HISTORY_DAYS = 365
LOOKBACK = 7

# Fetch sentiment score
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

# Fetch and prepare stock data with indicators and label
def fetch_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=False)
    if df.empty:
        return None

    df = df.reset_index()
    df.columns = df.columns.str.lower()
    df['ticker'] = ticker
    df['mood_score'] = fetch_sentiment_score(ticker)

    # Add technical indicators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    df['sma_10'] = ta.sma(df['close'], length=10)
    df['sma_50'] = ta.sma(df['close'], length=50)

    # Create target label: price goes up by >2% in next 5 days
    df['target'] = (df['close'].shift(-5) > df['close'] * 1.02).astype(int)

    df = df.dropna()
    return df

# Prepare training data
X_all, y_all = [], []
for ticker in TICKERS:
    df = fetch_stock_data(ticker)
    if df is not None:
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'mood_score', 'rsi', 'macd', 'sma_10', 'sma_50']
        data = df[feature_cols].values
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        X = [data[i-LOOKBACK:i] for i in range(LOOKBACK, len(data))]
        y = df['target'].values[LOOKBACK:]
        if len(X) == len(y):
            X_all.append(np.array(X))
            y_all.append(np.array(y))

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 10)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_all, y_all, epochs=20, batch_size=32, validation_split=0.2)

# Save model
model.save("stock_predictor_10feature.h5")