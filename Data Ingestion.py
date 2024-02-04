import yfinance as yf

def ingest_data(symbol, period, interval):
    """
    Fetch historical market data
    """
    data = yf.download(symbol, period=period, interval=interval)
    return data

