import pandas as pd

def clean_stock_data():
    # Read CSV while keeping the header structure
    raw_df = pd.read_csv("stock_data.csv", header=None)
    
    # Extract tickers and price types from first two rows
    tickers = raw_df.iloc[0, 1:].values
    price_types = raw_df.iloc[1, 1:].values
    
    # Create proper column names
    columns = ['date'] + [f"{ticker}_{ptype}" 
                         for ticker, ptype in zip(tickers, price_types)]
    
    # Read data again with proper columns (skip first 3 rows)
    stock_df = pd.read_csv("stock_data.csv", 
                         header=None,
                         skiprows=3,
                         names=columns)
    
    # Melt to long format
    melted_df = pd.melt(stock_df,
                       id_vars=['date'],
                       var_name='ticker_price',
                       value_name='value')
    
    # Split combined column
    melted_df[['ticker', 'price_type']] = melted_df['ticker_price'].str.split('_', expand=True)
    
    # Pivot to final structure
    clean_df = melted_df.pivot(index=['date', 'ticker'],
                             columns='price_type',
                             values='value').reset_index()
    
    return clean_df

# Load sentiment data
sentiment_dfs = []
for ticker in ['AAPL', 'GOOGL', 'MSFT']:
    try:
        news_df = pd.read_csv(f"news_{ticker.lower()}_with_mood.csv", parse_dates=['publishedAt'])
        news_df['date'] = news_df['publishedAt'].dt.date
        news_df['ticker'] = ticker  # Add ticker column
        sentiment_dfs.append(news_df)
    except FileNotFoundError:
        print(f"⚠️ No sentiment file found for {ticker}")

all_sentiment = pd.concat(sentiment_dfs)
daily_mood = all_sentiment.groupby(['date', 'ticker'])['mood_score'].mean().reset_index()

# Merge sentiment with ALL stocks
stock_df = clean_stock_data()
stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date

merged_data = pd.merge(stock_df, daily_mood, on=['date', 'ticker'], how='left')

# Save
merged_data.to_csv("final_dataset_ALL.csv", index=False)
print("✅ Success! Merged data for ALL stocks saved.")