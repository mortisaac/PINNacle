import pandas as pd
import torch
import os
import json

from models import PriceNetwork, VolatilityNetwork
from training import train_pinn
from data_utils import fetch_and_process_data
from plotting import (
    plot_price_surface,
    plot_local_vol_surface,
    calculate_implied_vol_surface,
    plot_implied_vol_surface
)
from utils import get_normalization_stats, normalize_inputs

def main():
    """
    Main function to run the training or inference pipeline.
    """
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ask user for ticker
    ticker = input("Enter the stock ticker (e.g., SPY, AAPL): ").upper()

    # Ask user for action
    choice = ''
    while choice.lower() not in ['y', 'n']:
        choice = input(f"Retrain the model for {ticker}? (y/n): ")
    
    if choice.lower() == 'y':
        run_training(device, ticker)
    elif choice.lower() == 'n':
        run_inference(device, ticker)

def run_training(device, ticker="SPY"):
    """Handles the model training path."""
    weights_path = f'pinn_{ticker.lower()}_weights.pth'
    if os.path.exists(weights_path):
        print(f"Found existing weights at '{weights_path}'. Skipping training.")
        return

    try:
        print(f"\n--- Starting Retraining Process for {ticker} ---")
        # Step 1: Fetch live market data.
        stock_price, vix, irx = fetch_and_process_data(ticker)

        # Print the fetched stock price, VIX, and T-bill rate.
        print(f"{ticker} Price: {stock_price:.2f}")
        print(f"VIX: {vix:.2f}")
        print(f"13-Week T-bill Rate: {irx:.4f}")

        # Save market parameters for later use
        params_path = f'market_params_{ticker.lower()}.json'
        market_params = {'stock_price': stock_price, 'vix': vix, 'irx': irx}
        with open(params_path, 'w') as f:
            json.dump(market_params, f)
        print(f"Saved market parameters to {params_path}")

        # Step 2: Load data and train the model
        data_path = f'{ticker.lower()}_option_data.csv'
        df_all_options = pd.read_csv(data_path)

        # --- Data Filtering ---
        is_itm_put = (df_all_options['Type'] == 'put') & (df_all_options['Strike'] > stock_price * 1.02)
        is_deep_itm_call = (df_all_options['Type'] == 'call') & (df_all_options['Strike'] < stock_price * 0.9)
        
        df_filtered = df_all_options[~is_itm_put & ~is_deep_itm_call]
        print(f"Filtered out {len(df_all_options) - len(df_filtered)} ITM options. Using {len(df_filtered)} options for training.")
        
        input_min, input_max = get_normalization_stats(df_all_options, stock_price)
        input_min, input_max = input_min.to(device), input_max.to(device)

        price_model = PriceNetwork().to(device)
        vol_model = VolatilityNetwork().to(device)

        train_pinn(price_model, vol_model, df_filtered, S_fixed=stock_price, r_fixed=irx, device=device, epochs=5000, pretrain_epochs=2000, lr=0.0005, input_min=input_min, input_max=input_max)

        # Step 3: Save the trained model's state
        torch.save({
            'price_model_state_dict': price_model.state_dict(),
            'vol_model_state_dict': vol_model.state_dict(),
            'input_min': input_min,
            'input_max': input_max
        }, weights_path)
        print(f"\nSuccessfully saved trained model weights to {weights_path}")

    except Exception as e:
        print(f"An error occurred during training for {ticker}: {e}")

def run_inference(device, ticker="SPY"):
    """Handles the model inference path."""
    print(f"\n--- Using Existing Model for Inference on {ticker} ---")
    weights_path = f'pinn_{ticker.lower()}_weights.pth'
    params_path = f'market_params_{ticker.lower()}.json'
    data_path = f'{ticker.lower()}_option_data.csv'

    if not all(os.path.exists(p) for p in [weights_path, params_path, data_path]):
        print(f"Error: Cannot find required files for {ticker}.")
        print("Please run the script with the retraining option 'y' first.")
        return

    with open(params_path, 'r') as f:
        market_params = json.load(f)
    stock_price = market_params.get('stock_price')
    print(f"Loaded market parameters: {ticker} Price={stock_price:.2f}, T-bill Rate={market_params['irx']:.4f}")

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
    print(f"Successfully loaded model weights from {weights_path}")

    # Example prediction
    S_sample = torch.tensor([[stock_price]], device=device)
    K_sample = torch.tensor([[stock_price]], device=device)
    T_sample = torch.tensor([[30/365.25]], device=device)
    
    price_input_sample_norm = normalize_inputs(S_sample, K_sample, T_sample, input_min, input_max)
    vol_input_sample_norm = price_input_sample_norm[:, [0, 2]]

    with torch.no_grad():
        predicted_price = price_model(price_input_sample_norm)
        predicted_local_vol = vol_model(vol_input_sample_norm)
    
    print(f"\nSample Prediction for {ticker}:")
    print(f"  - For S=${S_sample.item():.2f}, K=${K_sample.item():.2f}, T={T_sample.item()*365:.1f} days:")
    print(f"  - Predicted Price: ${predicted_price.item():.2f}")
    print(f"  - Predicted Local Vol: {predicted_local_vol.item():.4f}")

    print("\nCalculating Implied Volatility surface for plotting...")
    iv_surface, k_grid_iv, t_grid_iv, iv_min, iv_max = calculate_implied_vol_surface(
        price_model, df_all_options, stock_price, market_params['irx'], device, input_min, input_max
    )

    print("Generating plots...")
    price_fig = plot_price_surface(price_model, df_all_options, stock_price, device, input_min, input_max)
    price_fig.update_layout(title=f"Learned Option Price Surface for {ticker}")
    price_fig.show()

    lv_fig, _, _ = plot_local_vol_surface(vol_model, df_all_options, stock_price, device, input_min, input_max, iv_min, iv_max)
    lv_fig.update_layout(title=f"Learned Local Volatility Surface for {ticker}")
    lv_fig.show()

    iv_fig = plot_implied_vol_surface(iv_surface, k_grid_iv, t_grid_iv)
    iv_fig.update_layout(title=f"Learned Implied Volatility Surface for {ticker}")
    iv_fig.show()
    print("\nInference complete. Plots have been displayed.")

if __name__ == "__main__":
    main()