import pandas as pd
from textblob import TextBlob
import os

def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity  # Returns -1 (negative) to +1 (positive)

# Process ALL tickers
tickers = ['AAPL', 'GOOGL', 'MSFT']

for ticker in tickers:
    input_file = f"news_{ticker.lower()}.csv"
    output_file = f"news_{ticker.lower()}_with_mood.csv"
    
    if os.path.exists(input_file):
        print(f"🔍 Analyzing sentiment for {ticker}...")
        news = pd.read_csv(input_file)
        news['mood_score'] = news['title'].apply(analyze_sentiment)
        news.to_csv(output_file, index=False)
        print(f"✅ Saved {output_file}")
    else:
        print(f"⚠️ File not found: {input_file}")

print("\n🎉 All sentiment files generated!")