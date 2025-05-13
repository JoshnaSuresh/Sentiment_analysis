import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Configuration
STOCK_TICKERS = ['AAPL', 'MSFT', 'GOOGL']  # Example stocks
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data using yfinance"""
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def fetch_news_data(stock_name):
    """Fetch financial news using NewsAPI"""
    API_KEY = os.getenv('NEWS_API_KEY')
    url = f"https://newsapi.org/v2/everything?q={stock_name}&apiKey={API_KEY}"
    response = requests.get(url)
    return response.json()

def save_data(data, filename):
    """Save data to CSV"""
    if isinstance(data, pd.DataFrame):
        data.to_csv(filename)
    else:
        pd.DataFrame(data).to_csv(filename)

# Main execution
if __name__ == "__main__":
    # Fetch stock data
    stock_data = fetch_stock_data(STOCK_TICKERS, START_DATE, END_DATE)
    save_data(stock_data, 'stock_data.csv')
    
    # Fetch news data for each stock
    for ticker in STOCK_TICKERS:
        news_data = fetch_news_data(ticker)
        save_data(news_data['articles'], f'news_{ticker.lower()}.csv')
    
    print("✅ Data collection complete! Check your folder for CSV files.")