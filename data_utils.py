import yfinance as yf
import pandas as pd
from datetime import date

def fetch_and_process_data(ticker="SPY"):
    """
    Fetches stock and options data for a given ticker, processes it, and saves it to a CSV file.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., "SPY", "^SPX").

    Returns:
        tuple: A tuple containing stock_price, vix, and irx.
    """
    # 1. Fetches the current stock price for the given ticker.
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period="1d")
    if stock_history.empty:
        raise ValueError(f"Could not fetch historical price for '{ticker}'. It may be an invalid ticker or an index (try adding '^', e.g., '^SPX').")
    stock_price = stock_history['Close'].iloc[-1]

    # Fetch VIX and T-bill rate (market-wide indicators)
    vix_ticker = yf.Ticker("^VIX")
    vix_history = vix_ticker.history(period="1d")
    if vix_history.empty:
        # This is unlikely to fail but good practice to check
        raise ValueError("Could not fetch VIX data (^VIX). Market data may be unavailable.")
    vix = vix_history['Close'].iloc[-1]
    
    irx_ticker = yf.Ticker("^IRX")
    irx_history = irx_ticker.history(period="1d")
    if not irx_history.empty:
        irx = irx_history['Close'].iloc[-1] / 100 # Convert from percentage to decimal
    else:
        print("Could not fetch 13-week T-bill rate (^IRX). Setting to 0.")
        irx = 0.0

    # Get all available expiry dates
    if not stock.options:
        raise ValueError(f"No option expiry dates found for '{ticker}'. The ticker may not have options or is invalid.")
    all_expiries = stock.options
    all_options_data = []

    def process_options(options_df, option_type, time_to_expiry):
        df = options_df[(options_df['volume'] > 0) & (options_df['bid'] > 0) & (options_df['ask'] > 0)].copy()
        df['MarketPrice'] = (df['bid'] + df['ask']) / 2
        df = df[df['MarketPrice'] > 0.10]
        df['Type'] = option_type
        df['TimeToExpiry'] = time_to_expiry
        return df[['strike', 'Type', 'MarketPrice', 'TimeToExpiry']].rename(columns={'strike': 'Strike'})

    # --- Resilient Data Fetching Loop ---
    num_expiries_to_fetch = 3
    for expiry in all_expiries:
        print(f"Fetching data for {ticker} expiry: {expiry}")
        
        expiry_date = date.fromisoformat(expiry)
        today = date.today()
        time_to_expiry = (expiry_date - today).days / 365.25

        if time_to_expiry <= 0:
            print(f"Skipping past or same-day expiry: {expiry}")
            continue

        try:
            option_chain = stock.option_chain(expiry)
        except Exception as e:
            print(f"Could not fetch option chain for {expiry}. Error: {e}")
            continue
        
        expiry_had_data = False
        calls_df = process_options(option_chain.calls, 'call', time_to_expiry)
        puts_df = process_options(option_chain.puts, 'put', time_to_expiry)

        if not calls_df.empty:
            all_options_data.append(calls_df)
            expiry_had_data = True
        if not puts_df.empty:
            all_options_data.append(puts_df)
            expiry_had_data = True

        if expiry_had_data:
            num_expiries_to_fetch -= 1
            if num_expiries_to_fetch == 0:
                break

    if not all_options_data:
        raise ValueError(
            f"No option data found for {ticker} that meets the filtering criteria. "
            "This can happen for illiquid tickers or outside of market hours."
        )
    
    final_df = pd.concat(all_options_data)
    
    # Save the combined DataFrame to a ticker-specific CSV file
    output_filename = f'{ticker.lower()}_option_data.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"Successfully saved multi-expiry option data to {output_filename}")
    
    return stock_price, vix, irx