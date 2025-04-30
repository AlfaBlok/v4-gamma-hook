import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Define constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ETH_DATA_PATH = os.path.join(DATA_DIR, "eth_historical.csv")

def fetch_historical_data(symbol="ETH-USD", period="max", interval="1d", cache=True):
    """
    Fetch historical price data for a given symbol from Yahoo Finance.
    
    Args:
        symbol (str): The ticker symbol to fetch data for (default: "ETH-USD")
        period (str): The time period to fetch data for (default: "max")
        interval (str): The data interval (default: "1d" for daily)
        cache (bool): Whether to cache the data locally (default: True)
        
    Returns:
        pandas.DataFrame: Historical price data
    """
    try:
        # Try to load from cache first if caching is enabled
        if cache and os.path.exists(ETH_DATA_PATH):
            cached_data = load_cached_data()
            
            # Check if cache loaded successfully and has data
            if not cached_data.empty and isinstance(cached_data.index, pd.DatetimeIndex):
                # Make sure we're using naive datetimes for comparison
                last_date = cached_data.index[-1]
                if isinstance(last_date, datetime):
                    now = datetime.now()
                    # Convert both to naive if either has timezone
                    if last_date.tzinfo is not None:
                        last_date = last_date.replace(tzinfo=None)
                    if now.tzinfo is not None:
                        now = now.replace(tzinfo=None)
                        
                    if (now - last_date).days < 1:
                        print(f"Using cached data (last updated: {last_date})")
                        return cached_data
        
        # Fetch fresh data if cache doesn't exist or is outdated
        print(f"Fetching {symbol} data from Yahoo Finance...")
        
        # Fix: Handle the ticker properly for yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        # Ensure data loaded properly
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Ensure consistent timezone handling - convert to naive datetimes
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tzinfo is not None:
            data.index = data.index.tz_localize(None)
        
        # Ensure numeric data types for price columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Cache the data if caching is enabled
        if cache:
            # Ensure the data directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
            data.to_csv(ETH_DATA_PATH)
            print(f"Data cached to {ETH_DATA_PATH}")
        
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        
        # If fetching fails but cache exists, use cached data
        if cache and os.path.exists(ETH_DATA_PATH):
            print("Falling back to cached data...")
            return load_cached_data()
        
        # If all else fails, return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])

def load_cached_data():
    """
    Load historical price data from local cache.
    
    Returns:
        pandas.DataFrame: Historical price data
    """
    try:
        # Check if file exists
        if not os.path.exists(ETH_DATA_PATH):
            print(f"Cached data file not found: {ETH_DATA_PATH}")
            return pd.DataFrame()
            
        # Read CSV with robust date parsing
        data = pd.read_csv(
            ETH_DATA_PATH,
            index_col=0,
            parse_dates=True
        )
        
        # Skip first row if it contains headers instead of data
        if "Ticker" in str(data.index[0]) or not pd.api.types.is_datetime64_any_dtype(data.index):
            print("Header row detected in CSV, skipping first row")
            data = pd.read_csv(
                ETH_DATA_PATH,
                index_col=0,
                parse_dates=True,
                skiprows=1
            )
        
        # Double-check that index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, errors='coerce')
            # Drop rows where index couldn't be parsed
            data = data[~data.index.isna()]
        
        # Ensure consistent timezone handling - convert to naive datetimes
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tzinfo is not None:
            data.index = data.index.tz_localize(None)
            
        # Ensure numeric data types for price columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN Close prices
        data = data.dropna(subset=['Close'])
        
        return data
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]) 