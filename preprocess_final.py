import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load data
df = pd.read_csv("final_dataset_ALL.csv")

# 2. Handle missing values
df['mood_score'] = df['mood_score'].fillna(0)

# 3. Verify columns exist
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'mood_score']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# 4. Normalize features PER TICKER
features = required_cols.copy()
for ticker in df['ticker'].unique():
    mask = df['ticker'] == ticker
    df.loc[mask, features] = MinMaxScaler().fit_transform(df.loc[mask, features])

# 5. Create sequences
def create_sequences(data, window=7):
    X, y = [], []
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker].dropna()
        for i in range(window, len(ticker_data)):
            X.append(ticker_data.iloc[i-window:i][features].values)
            y.append(1 if ticker_data.iloc[i]['Close'] > ticker_data.iloc[i-1]['Close'] else 0)
    return np.array(X), np.array(y)

X, y = create_sequences(df)
np.savez("training_data.npz", X=X, y=y)
print(f"✅ Success! Created {len(X)} training samples.")