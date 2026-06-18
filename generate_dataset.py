import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def download_and_split_data(ticker="AAPL", start_date="2018-01-01", end_date="2025-01-01"):
    print(f"📡 Fetching real historical data for {ticker} from Yahoo Finance...")
    os.makedirs("data/processed", exist_ok=True)
    
    # Download real historical entries
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError("❌ Failed to download data. Check connection or symbol.")
        
    # Isolate relevant structural baseline dimensions
    df = data[['Close', 'Open', 'High', 'Low', 'Volume']].copy()
    
    # Apply standard Scaling MinMax normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df)
    
    scaled_df = pd.DataFrame(
        scaled_features, 
        columns=['Close', 'Open', 'High', 'Low', 'Volume'],
        index=df.index
    )
    
    # Calculate chronological division points (70% Train, 15% Val, 15% Test)
    n = len(scaled_df)
    train_df = scaled_df[0:int(n*0.7)]
    val_df = scaled_df[int(n*0.7):int(n*0.85)]
    test_df = scaled_df[int(n*0.85):]
    
    # Export explicitly to paths
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"📊 Dataset processing complete! Loaded {n} total daily market candles.")
    print("💾 Files successfully written: data/processed/[train, val, test].csv")

if __name__ == "__main__":
    download_and_split_data()