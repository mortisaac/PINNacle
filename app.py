import streamlit as st
import torch
import yfinance as yf
import pandas as pd
import os
import json
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from newsapi import NewsApiClient
from transformers import pipeline

# Import necessary components from your existing modules
from models import PriceNetwork, VolatilityNetwork
from plotting import (
    plot_price_surface,
    plot_local_vol_surface,
    calculate_implied_vol_surface,
    plot_implied_vol_surface
)
from utils import calculate_greeks, normalize_inputs, get_normalization_stats
from training import train_pinn
from data_utils import fetch_and_process_data

@st.cache_resource
def load_sentiment_pipeline():
    """
    Loads the sentiment analysis pipeline from Hugging Face.
    """
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_genai_model():
    """
    Loads the Gemini Pro model.
    """
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        return genai.GenerativeModel('gemini-flash-latest')
    except (FileNotFoundError, KeyError):
        return None

@st.cache_data(ttl=1800)
def fetch_live_news(api_key, ticker, long_name):
    """
    Fetches live financial news headlines using the 'everything' endpoint for a broader search.
    It prioritizes recent news containing the company's name or ticker.
    """
    if not api_key or api_key == "YOUR_NEWS_API_KEY_HERE":
        return None, "NewsAPI key not configured.", False

    try:
        newsapi = NewsApiClient(api_key=api_key)
        is_fallback = False

        # --- Tier 1: Search for the specific company ---
        # Construct a robust query. Searching for the long name in quotes gives it priority.
        # We also add keywords to focus the search on relevant topics.
        query = f'("{long_name}" OR {ticker}) AND (business OR finance OR technology OR stocks)'
        print(f"Fetching news with query: '{query}'")
        
        # Use get_everything for a broader search across all indexed sources, sorted by recency.
        all_articles_response = newsapi.get_everything(q=query, language='en', sort_by='relevancy')
        articles = all_articles_response.get('articles', [])

        # --- Tier 2: Fallback to general business news if no specific news is found ---
        if not articles:
            print(f"No specific articles found for '{query}'. Falling back to general business news.")
            is_fallback = True
            # Use top_headlines for the fallback to get major business news.
            fallback_response = newsapi.get_top_headlines(category='business', language='en', country='us')
            articles = fallback_response.get('articles', [])

        if articles:
            formatted_articles = [{"source": article['source']['name'], "headline": article['title']} for article in articles[:7]]
            return formatted_articles, None, is_fallback
        
        return None, f"No recent articles found for {ticker}.", is_fallback

    except Exception as e:
        return None, f"Failed to fetch news: {e}", False

def run_training(device, ticker):
    """Handles the model training path, checking for existing files first."""
    weights_path = f'pinn_{ticker.lower()}_weights.pth'
    params_path = f'market_params_{ticker.lower()}.json'
    data_path = f'{ticker.lower()}_option_data.csv'

    # If all required files for the given ticker already exist, skip training.
    if all(os.path.exists(p) for p in [weights_path, params_path, data_path]):
        st.toast(f"Found existing model and data for {ticker}. Skipping training.", icon="📁")
        print(f"Found existing files for {ticker}. Skipping training.")
        return True

    # --- One-time migration for legacy SPY files ---
    # If the new params file doesn't exist but the old one does, rename it.
    if ticker.upper() == 'SPY' and not os.path.exists(params_path) and os.path.exists('market_params.json'):
        print("Migrating legacy 'market_params.json' to 'market_params_spy.json'")
        try:
            os.rename('market_params.json', params_path)
            # If the other files also exist, we can now skip training.
            if all(os.path.exists(p) for p in [weights_path, params_path, data_path]):
                st.toast(f"Migrated legacy files for SPY. Skipping training.")
                return True
        except OSError as e:
            st.warning(f"Could not migrate legacy 'market_params.json'. Proceeding with retraining. Error: {e}")

    # If files are missing, proceed with the full training process.
    try:
        with st.spinner(f"Starting training process for {ticker}... This may take a while."):
            print(f"\n--- Starting Retraining Process for {ticker} ---")
            stock_price, vix, irx = fetch_and_process_data(ticker)

            st.toast(f"Fetched Data: {ticker} @ ${stock_price:.2f} | VIX: {vix:.2f}", icon="📈")

            market_params = {'stock_price': stock_price, 'vix': vix, 'irx': irx}
            with open(params_path, 'w') as f:
                json.dump(market_params, f)
            print(f"Saved market parameters to {params_path}")

            df_all_options = pd.read_csv(data_path)

            is_itm_put = (df_all_options['Type'] == 'put') & (df_all_options['Strike'] > stock_price * 1.02)
            is_deep_itm_call = (df_all_options['Type'] == 'call') & (df_all_options['Strike'] < stock_price * 0.9)
            df_filtered = df_all_options[~is_itm_put & ~is_deep_itm_call]
            print(f"Filtered out {len(df_all_options) - len(df_filtered)} ITM options.")

            input_min, input_max = get_normalization_stats(df_all_options, stock_price)
            input_min, input_max = input_min.to(device), input_max.to(device)

            price_model = PriceNetwork().to(device)
            vol_model = VolatilityNetwork().to(device)

            train_pinn(price_model, vol_model, df_filtered, S_fixed=stock_price, r_fixed=irx, device=device, epochs=5000, pretrain_epochs=2000, lr=0.0005, input_min=input_min, input_max=input_max)

            torch.save({
                'price_model_state_dict': price_model.state_dict(),
                'vol_model_state_dict': vol_model.state_dict(),
                'input_min': input_min,
                'input_max': input_max
            }, weights_path)
            print(f"\nSuccessfully saved trained model weights to {weights_path}")
            st.toast(f"Training for {ticker} complete!", icon="✅")
            return True
    except Exception as e:
        st.error(f"An error occurred during training for {ticker}: {e}")
        print(f"An error occurred during training for {ticker}: {e}")
        return False

@st.cache_data
def load_data_and_models(ticker):
    """
    Loads models, market parameters, and option data from disk for a given ticker.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = f'pinn_{ticker.lower()}_weights.pth'
    params_path = f'market_params_{ticker.lower()}.json'
    data_path = f'{ticker.lower()}_option_data.csv'

    if not all(os.path.exists(p) for p in [weights_path, params_path, data_path]):
        return None, None, None, None, None, None, None, None

    with open(params_path, 'r') as f:
        market_params = json.load(f)
    
    df_all_options = pd.read_csv(data_path)

    price_model = PriceNetwork().to(device)
    vol_model = VolatilityNetwork().to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    price_model.load_state_dict(checkpoint['price_model_state_dict'])
    vol_model.load_state_dict(checkpoint['vol_model_state_dict'])
    input_min = checkpoint['input_min'].to(device)
    input_max = checkpoint['input_max'].to(device)
    
    price_model.eval()
    vol_model.eval()

    yf_ticker = yf.Ticker(ticker)
    hist = yf_ticker.history(period="30d")['Close']
    log_returns = np.log(hist / hist.shift(1))
    realized_std = log_returns.iloc[-20:].std()
    realized_vol = realized_std * np.sqrt(252)

    return price_model, vol_model, df_all_options, market_params, input_min, input_max, device, realized_vol, hist

@st.cache_data
def generate_surfaces(_price_model, _vol_model, _df_options, stock_price, r_fixed, _input_min, _input_max, device, realized_vol, ticker):
    """
    Calculates the IV and LV surfaces.
    """
    iv_surface, k_grid_iv, t_grid_iv, iv_min, iv_max = calculate_implied_vol_surface(
        _price_model, _df_options, stock_price, r_fixed, device, _input_min, _input_max
    )
    price_fig = plot_price_surface(_price_model, _df_options, stock_price, device, _input_min, _input_max)
    lv_fig, lv_skew, lv_tilt = plot_local_vol_surface(_vol_model, _df_options, stock_price, device, _input_min, _input_max, iv_min, iv_max)
    iv_fig = plot_implied_vol_surface(iv_surface, k_grid_iv, t_grid_iv)

    k_range = k_grid_iv[:, 0].cpu().numpy()
    k_atm_idx = np.argmin(np.abs(k_range - stock_price))
    
    atm_iv_slice = iv_surface[k_atm_idx, :]
    avg_atm_iv = np.nanmean(atm_iv_slice)

    vol_spread = (avg_atm_iv - realized_vol) * 100 if not np.isnan(avg_atm_iv) else np.nan

    S_greek = torch.tensor([stock_price], device=device, dtype=torch.float32, requires_grad=True)
    K_greek = torch.tensor([stock_price], device=device, dtype=torch.float32)
    T_greek = torch.tensor([30/365.25], device=device, dtype=torch.float32, requires_grad=True)
    atm_price, _, _, atm_theta = calculate_greeks(_price_model, S_greek, K_greek, T_greek, _input_min, _input_max)

    is_itm_put = (_df_options['Type'] == 'put') & (_df_options['Strike'] > stock_price * 1.02)
    is_deep_itm_call = (_df_options['Type'] == 'call') & (_df_options['Strike'] < stock_price * 0.9)
    df_filtered = _df_options[~is_itm_put & ~is_deep_itm_call]

    S_data = torch.full((len(df_filtered), 1), stock_price, device=device)
    K_data = torch.tensor(df_filtered['Strike'].values, dtype=torch.float32).unsqueeze(1).to(device)
    T_data = torch.tensor(df_filtered['TimeToExpiry'].values, dtype=torch.float32).unsqueeze(1).to(device)
    market_price = torch.tensor(df_filtered['MarketPrice'].values, dtype=torch.float32).unsqueeze(1).to(device)

    price_inputs_data_norm = normalize_inputs(S_data, K_data, T_data, _input_min, _input_max)
    with torch.no_grad():
        C_hat_data = _price_model(price_inputs_data_norm)
    
    model_fit_error = torch.mean(torch.abs(C_hat_data - market_price)).item()

    return price_fig, lv_fig, iv_fig, lv_skew, lv_tilt, avg_atm_iv, atm_price, atm_theta, vol_spread, model_fit_error

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(layout="wide", page_title="PINN-acle Option Analysis")

    # --- 1. Implement Custom CSS for Title and Market Data ---
    st.markdown("""<style>
        div.block-container {
            padding-top: 1.5rem;
        }
        .custom-title {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
            font-size: 3.5rem; 
            font-weight: bold;
            color: #E2E8F0;
            margin-bottom: -0.5rem;
            display: flex;
            align-items: center;
        }
        .pinn-highlight {
            background: linear-gradient(90deg, #3B82F6, #A855F7);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            display: inline-block;
            font-size: 1.15em;
            font-weight: 900;
        }
        .title-middle { position: relative; top: 0.0em; }
        .title-suffix {
            font-size: 0.7em;
            color: #94A3B8;
            margin-left: 0.7rem;
            opacity: 0.7;
            position: relative;
            top: 0.15em;
        }
        .market-context { font-size: 1.5rem; color: #94A3B8; text-align: left; margin-bottom: 1.5rem; }
        .market-context b { font-weight: 700; color: #E2E8F0; }
        .help-icon-container { position: relative; display: inline-block; margin-left: 15px; cursor: help; top: 0.5rem; }
        .help-icon {
            font-size: 1.5rem; font-weight: bold; color: #6B7280; border: 2px solid #6B7280;
            border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
        }
        .summary-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #E2E8F0;
            display: flex; align-items: center;
            margin-bottom: 1rem;
        }
        .simple-summary-box {
            border: 1px solid #4A5568;   /* A neutral gray border */
            border-radius: 10px;        /* Rounded corners */
            padding: 1rem;              /* Padding inside the box */
            margin-top: 1rem;           /* Use a positive margin to push the box down */
            background-color: #1a1f2b;   /* A slightly different background to stand out */
            max-width: 95%;             /* Prevent it from taking full column width */
            min-height: 150px;          /* Ensure a consistent minimum height */
            font-size: 1.1rem;         /* Increase body text size inside the box */
            white-space: pre-wrap;      /* Preserve line breaks from the AI response */
        }
        .help-icon {
            font-size: 1.5rem; font-weight: bold; color: #6B7280; border: 2px solid #6B7280;
            border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
        }
        /* --- CSS for Metric Tooltips --- */
        .metric-container { position: relative; }
        .metric-label {
            font-size: 1rem; color: #94A3B8; display: flex; align-items: center; gap: 8px;
        }
        .metric-value {
            font-size: 1.75rem; font-weight: 600; color: #FAFAFA; padding-top: 0.25rem;
        }
        .metric-help-icon {
            font-size: 0.7rem; font-weight: bold; color: #6B7280; border: 1px solid #6B7280;
            border-radius: 50%; width: 16px; height: 16px; display: flex; align-items: center; justify-content: center;
            cursor: help;
        }
        .metric-tooltip-text {
            visibility: hidden; width: 300px; background-color: #1a1f2b; color: #FAFAFA; text-align: left;
            border-radius: 6px; padding: 10px; border: 1px solid #4A5568; position: absolute; z-index: 1;
            bottom: 125%; left: 50%; transform: translateX(-50%); opacity: 0; transition: opacity 0.3s;
            font-size: 0.85rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 400;
        }
        .metric-container:hover .metric-tooltip-text {
            visibility: visible; opacity: 1;
        }
        /* --- End of Metric Tooltips CSS --- */
        .tooltip-text {
            visibility: hidden; width: 450px; background-color: #1a1f2b; color: #FAFAFA; text-align: left;
            border-radius: 6px; padding: 10px; border: 1px solid #4A5568; position: absolute; z-index: 1;
            top: 120%; left: 50%; transform: translateX(-50%); opacity: 0; transition: opacity 0.3s;
            font-size: 0.85rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 400;
        }
        .help-icon-container:hover .tooltip-text { visibility: visible; opacity: 1; }
        .positive-change { color: #48BB78; font-weight: 600; }
        .negative-change { color: #F56565; font-weight: 600; }
    </style>""", unsafe_allow_html=True)

    if 'ticker' not in st.session_state:
        st.session_state.ticker = 'SPY'
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False

    # --- Header and Ticker Input ---
    header_col1, header_col2 = st.columns([0.7, 0.3])
    with header_col1:
        help_text = (
            "<b>PINN-acle</b> is an advanced financial analysis tool that provides a clear, quantitative snapshot of market risk and fear. It goes beyond simple observation by building a robust, physics-based model of option prices.<br><br>"
            "<b>What it does:</b> The tool ingests raw market option data, calculates key risk gauges (like Skew and Term Structure), tracks social sentiment, and synthesizes all signals into an expert, interpretive market summary.<br><br>"
            "<b>How PINNs Power It:</b> At its core, PINN-acle is run by a Physics-Informed Neural Network (PINN). Traditional A.I. learns only from data. A PINN is different: it's a special neural network that is also trained to obey the fundamental laws of finance—specifically, the Black-Scholes pricing equation. This forces the model to be mathematically consistent and arbitrage-free, allowing it to accurately extract the Local Volatility Surface, which is the true engine of option pricing and risk. This ensures the output is not just a guess, but a stable, trustworthy prediction rooted in financial theory."
        )
        title_with_help = f'''<div class="custom-title"><span class="pinn-highlight">PINN</span><span class="title-middle">-acle:</span> <span class="title-suffix">Option Analysis</span> <div class="help-icon-container"><span class="help-icon">?</span><div class="tooltip-text">{help_text}</div></div></div>'''
        st.markdown(title_with_help, unsafe_allow_html=True)

    # --- Define Callbacks for UI interaction ---
    def trigger_analysis():
        """Callback to set session state for analysis."""
        st.session_state.ticker = st.session_state.ticker_search_box.upper()
        st.session_state.analysis_triggered = True
        if 'llm_summary' in st.session_state:
            del st.session_state['llm_summary']

    def trigger_retrain():
        """Callback to trigger analysis and clear old files for retraining."""
        trigger_analysis() # First, set the state for analysis
        ticker_to_train = st.session_state.ticker
        # Force retraining by deleting existing model/data files
        for file_suffix in ['weights.pth', 'option_data.csv', 'params.json']:
            file_path = f'pinn_{ticker_to_train.lower()}_{file_suffix}'
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed existing file to force retraining: {file_path}")

    with header_col2:
        st.markdown('<div style="margin-top: 35px;"></div>', unsafe_allow_html=True) # Add precise vertical space
        input_col, analyze_col, retrain_col = st.columns([0.5, 0.25, 0.25])

        # The text input now triggers the analysis on its own when Enter is pressed
        input_col.text_input("Search Ticker", st.session_state.ticker, key="ticker_search_box",
                             label_visibility="collapsed", on_change=trigger_analysis)

        # The buttons now simply call their respective callback functions
        analyze_col.button('Analyze', use_container_width=True, on_click=trigger_analysis)
        retrain_col.button('(Re)-train', use_container_width=True, on_click=trigger_retrain)

    # --- Main Analysis Body ---
    if st.session_state.analysis_triggered:
        ticker = st.session_state.ticker
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if run_training(device, ticker):
            price_model, vol_model, df_all_options, market_params, input_min, input_max, device, realized_vol, price_history = load_data_and_models(ticker)
            
            if price_model is None:
                st.error(f"Could not load data for {ticker}. Please ensure training was successful or that the ticker is valid.")
                return

            stock_price = market_params.get('stock_price') or market_params.get('spy_price')
            r_fixed = market_params.get('irx')

            try:
                yf_ticker_obj = yf.Ticker(ticker)
                long_name = yf_ticker_obj.info.get('longName', ticker) # Fallback to ticker if longName not found
                hist = yf_ticker_obj.history(period="2d")
                if len(hist) >= 2:
                    current_price, previous_close = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                    change = current_price - previous_close
                    percent_change = (change / previous_close) * 100
                    change_color_class = "positive-change" if change >= 0 else "negative-change"
                    st.markdown(f'<div class="market-context"><b>{long_name} ({ticker})</b> @ ${current_price:.2f} <span class="{change_color_class}">{change:+.2f} ({percent_change:+.2f}%)</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="market-context"><b>{long_name} ({ticker})</b> @ ${stock_price:.2f}</div>', unsafe_allow_html=True)
            except Exception:
                 st.markdown(f'<div class="market-context"><b>{ticker}</b> @ ${stock_price:.2f}</div>', unsafe_allow_html=True)

            st.markdown('<hr style="margin-top: 1rem; margin-bottom: 0.75rem;">', unsafe_allow_html=True)

            price_fig, lv_fig, iv_fig, lv_skew, _, avg_atm_iv, atm_price, _, vol_spread, model_fit_error = generate_surfaces(
                price_model, vol_model, df_all_options, stock_price, r_fixed, input_min, input_max, device, realized_vol, ticker
            )

            metrics_col, history_plot_col = st.columns([0.55, 0.45])

            with metrics_col:
                st.subheader("Key Market & Model Metrics")
                st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True) # Add vertical space
                # First row of metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f"""<div class="metric-container">
                        <div class="metric-label">VIX Index <div class="metric-help-icon">?</div> <div class="metric-tooltip-text">The VIX Index is a real-time measure of the market's expectation of 30-day volatility. It is constructed using the prices of S&P 500 index options and is often called the 'fear gauge'.</div></div>
                        <div class="metric-value">{market_params.get('vix', 0):.2f}</div>
                    </div>""", unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"""<div class="metric-container">
                        <div class="metric-label">Realized Vol (20D) <div class="metric-help-icon">?</div> <div class="metric-tooltip-text">The historical volatility of the asset over the past 20 trading days. It measures how much the asset's price has actually moved.</div></div>
                        <div class="metric-value">{realized_vol*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f"""<div class="metric-container">
                        <div class="metric-label">Avg. ATM Implied Vol <div class="metric-help-icon">?</div> <div class="metric-tooltip-text">The average implied volatility for at-the-money options across all available expiries, as derived from the model's price surface. It represents the market's future volatility expectation.</div></div>
                        <div class="metric-value">{avg_atm_iv*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True) # Add vertical space

                # Second row of metrics
                m_colA, m_colB, m_colC = st.columns(3)
                with m_colA:
                    st.markdown(f"""<div class="metric-container">
                        <div class="metric-label">Vol Spread (Imp - Real) <div class="metric-help-icon">?</div> <div class="metric-tooltip-text">The difference between the average Implied Volatility and the 20-day Realized Volatility. A large positive spread indicates that options are pricing in more risk than has recently been observed.</div></div>
                        <div class="metric-value">{vol_spread:.1f} pts</div>
                    </div>""", unsafe_allow_html=True)
                with m_colB:
                    st.markdown(f"""<div class="metric-container">
                        <div class="metric-label">LV Skew Intensity <div class="metric-help-icon">?</div> <div class="metric-tooltip-text">Measures the steepness of the volatility smirk from the Local Volatility surface. A higher value indicates greater demand for downside protection (puts), reflecting fear of a market drop.</div></div>
                        <div class="metric-value">{lv_skew:.2f} pts</div>
                    </div>""", unsafe_allow_html=True)
                with m_colC:
                    st.markdown(f"""<div class="metric-container">
                        <div class="metric-label">ATM 30-Day Price <div class="metric-help-icon">?</div> <div class="metric-tooltip-text">The model's calculated price for a hypothetical at-the-money option with exactly 30 days to expiry. It serves as a standardized benchmark for option cost.</div></div>
                        <div class="metric-value">${atm_price:.2f}</div>
                    </div>""", unsafe_allow_html=True)
            
            with history_plot_col:
                st.subheader(f"{ticker} Price (30-Day)")
                history_fig = go.Figure()
                history_fig.add_trace(go.Scatter(x=price_history.index, y=price_history.values, mode='lines', line=dict(color='#87CEEB', width=3)))
                history_fig.update_layout(
                    height=215, margin=dict(l=40, r=10, b=20, t=5),
                    xaxis=dict(showgrid=False, showticklabels=False, title=None),
                    yaxis=dict(showgrid=False, showticklabels=True, title=None, tickprefix='$'),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(history_fig, use_container_width=True, config={'displayModeBar': False})

            st.markdown('<div style="margin-top: -40px;"><hr></div>', unsafe_allow_html=True)

            col_summary, col_sentiment = st.columns([0.6, 0.4])
            with col_sentiment:
                st.subheader("Recent News Sentiment")
                sentiment_pipeline = load_sentiment_pipeline()
                news_api_key = st.secrets.get("NEWS_API_KEY")
                headlines, error_msg, is_fallback = fetch_live_news(news_api_key, ticker, long_name)
                if error_msg: st.warning(error_msg)
                if headlines:
                    if is_fallback:
                        st.caption(f"Showing general business news as no articles were found for **{ticker}**.")

                    for item in headlines:
                        result = sentiment_pipeline(item["headline"])[0]
                        confidence = result.get('score', 0)
                        emoji = "🟢" if result['label'] == 'POSITIVE' else "🔴"
                        st.write(f"{emoji} **{result['label']}** ({confidence:.0%}): {item['headline']} *({item['source']})*")
                else: st.info("No news headlines to display.")

            with col_summary:
                genai_model = load_genai_model()
                if genai_model is None:
                    st.warning("Google API key not found. Cannot generate summary.")
                else:
                    if 'llm_summary' not in st.session_state:
                        with st.spinner("AI strategist is analyzing the data..."):
                            headline_text = "\n".join([f"- {h['headline']}" for h in headlines]) if headlines else "No recent news available."
                            prompt = f"""
You are a senior quantitative risk strategist providing a comprehensive market briefing. Your task is to analyze the data for **{long_name} ({ticker})** below and deliver the final output as exactly **three (3) distinct paragraphs** of 2-3 sentences each. In your response, refer to the company by its name, '{long_name}'.

The paragraphs **must be separated by a single empty line** (a double line break) to ensure clear spacing. The output **must be pure plaintext**; DO NOT use any Markdown formatting (no bolding, no headings, no bullet points).

**Analysis Content Guidance:**

1.  **Paragraph 1 (Valuation & Structural Risk):** Synthesize the VIX, Realized Vol, Vol Spread, and Avg ATM IV. Determine the current volatility regime (e.g., elevated/suppressed) and assess whether optionality is cheap or expensive relative to realized movement. Identify the core structural risk being priced in (e.g., immediate crash fear, term risk) based on the Skew Intensity.
2.  **Paragraph 2 (Synthesis & Implication):** **CRITICALLY, YOU MUST DIRECTLY REFERENCE THE HEADLINES.** Assess whether the subjects of the positive and negative news (e.g., "CEO apology," "China Tariffs") justify the observed structural risk. Conclude with a clear strategic implication for portfolio risk management.

**Market Data (Quant):**
- VIX Index (Overall Fear): {market_params.get('vix', 0):.2f}
- Realized Vol (20D): {realized_vol*100:.1f}%
- Avg ATM IV: {avg_atm_iv*100:.1f}%
- Vol Spread (Implied - Realized): {vol_spread:.1f} pts
- LV Skew Intensity (Downside Fear): {lv_skew:.2f} pts

**Recent Headlines (LITERAL TEXT FOR ANALYSIS):**
{headline_text}

**Provide the three-paragraph analysis now (plaintext only, separated by empty lines):**
"""
                            response = genai_model.generate_content(prompt)
                            st.session_state['llm_summary'] = response.text
                    
                    summary_text = st.session_state.get('llm_summary', 'No summary available.')
                    gemini_logo_svg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='%238B5CF6'%3E%3Cpath d='M12 0L9 9l-9 3 9 3 3 9 3-9 9-3-9-3z'/%3E%3C/svg%3E"
                    title_html = f'<div class="summary-title"><img src="{gemini_logo_svg}" style="margin-right: 10px;">AI-Powered Market Summary</div>'
                    st.markdown(f'<div class="simple-summary-box">{title_html}{summary_text}</div>', unsafe_allow_html=True)

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Learned Option Price Surface")
                st.caption("This is the direct output of the price network. It shows the model's arbitrage-free price for options across all strikes (K) and expiries (T), forming the foundation for all other derived surfaces. The surface's curvature reflects the option's sensitivity to changes in strike and time.")
                st.plotly_chart(price_fig, use_container_width=True)
            with col2:
                st.subheader("Learned Implied Volatility Surface")
                st.caption("Derived from the price surface, this shows the market's expectation of future volatility. The 'smile' or 'smirk' shape is a key feature, indicating that out-of-the-money options have higher implied volatility, reflecting demand for tail-risk protection.")
                st.plotly_chart(iv_fig, use_container_width=True)
            with col3:
                st.subheader("Learned Local Volatility Surface")
                st.caption("This surface, σ(S, T), represents the instantaneous volatility of the underlying asset at a future time T, given it is at price S. It is the core output of the volatility network and ensures the model is consistent with the Black-Scholes PDE.")
                st.plotly_chart(lv_fig, use_container_width=True)
if __name__ == "__main__":
    main()
