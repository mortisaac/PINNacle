import torch
import numpy as np
import plotly.graph_objects as go
import scipy.optimize as opt

from utils import normalize_inputs, black_scholes_call_torch

def plot_price_surface(price_model, df_options, S_fixed, device, input_min, input_max):
    """
    Plots the learned option price surface C(K, T) at a fixed spot price.
    """
    # Create a grid for K and T from the data ranges
    k_min, k_max = df_options['Strike'].min(), df_options['Strike'].max()
    t_min, t_max = df_options['TimeToExpiry'].min(), df_options['TimeToExpiry'].max()
    k_range = torch.linspace(k_min, k_max, 50, device=device)
    t_range = torch.linspace(t_min, t_max, 50, device=device)
    k_grid, t_grid = torch.meshgrid(k_range, t_range, indexing='ij')

    # Flatten grids to create a batch for the model
    K_flat = k_grid.flatten().unsqueeze(1)
    T_flat = t_grid.flatten().unsqueeze(1)
    S_flat = torch.full_like(K_flat, S_fixed)
    
    # Normalize the grid inputs before prediction
    price_inputs_norm = normalize_inputs(S_flat, K_flat, T_flat, input_min, input_max)
    
    with torch.no_grad():
        price_flat = price_model(price_inputs_norm)
    
    price_surface = price_flat.reshape(k_grid.shape)

    fig = go.Figure(data=[go.Surface(z=price_surface.cpu().numpy(), 
                                     x=t_grid.cpu().numpy(), 
                                     y=k_grid.cpu().numpy())])
    fig.update_layout(scene=dict(
                          xaxis=dict(title='Time to Expiry (Years)', nticks=10),
                          yaxis=dict(title='Strike Price (K)'),
                          zaxis=dict(title='Option Price ($)'),
                          camera=dict(eye=dict(x=1.6, y=1.6, z=1.6)) # Zoom in slightly
                      ),
                      autosize=False,
                      height=500, # Make plots shorter
                      margin=dict(l=65, r=50, b=65, t=40), # Reduced top margin
                      scene_dragmode='turntable') # Allow turntable rotation
    return fig

def plot_local_vol_surface(vol_model, df_options, S_current, device, input_min, input_max, iv_min, iv_max):
    """
    Plots the learned local volatility surface sigma_LV(S, T).
    """
    # Create a grid for S and T. S is sampled around the current price.
    s_min, s_max = S_current * 0.8, S_current * 1.2 # Wider range for vol surface
    t_min, t_max = df_options['TimeToExpiry'].min(), df_options['TimeToExpiry'].max()
    s_range = torch.linspace(s_min, s_max, 50, device=device)
    t_range = torch.linspace(t_min, t_max, 50, device=device)
    s_grid, t_grid = torch.meshgrid(s_range, t_range, indexing='ij')

    # Flatten grids to create a batch for the model
    S_flat = s_grid.flatten().unsqueeze(1)
    T_flat = t_grid.flatten().unsqueeze(1)
    
    # Normalize the grid inputs before prediction
    # We need K for the full normalization, so we can use a dummy K (e.g., S_current)
    K_dummy = torch.full_like(S_flat, S_current)
    vol_inputs_norm = normalize_inputs(S_flat, K_dummy, T_flat, input_min, input_max)[:, [0, 2]]
    
    with torch.no_grad():
        vol_flat = vol_model(vol_inputs_norm)
    
    vol_surface = vol_flat.reshape(s_grid.shape)

    # --- Dynamic Post-processing Scaling using pre-calculated IV range ---
    print(f"Using dynamic IV range for scaling: [{iv_min:.4f}, {iv_max:.4f}]")

    current_min = torch.min(vol_surface)
    current_max = torch.max(vol_surface)
    epsilon = 1e-8

    scaled_vol_surface = ((vol_surface - current_min) / (current_max - current_min + epsilon)) * (iv_max - iv_min) + iv_min

    # --- Calculate Skew and Term Structure Metrics from the SCALED surface ---
    # Define reference points for metrics
    S_low, S_high = S_current * 0.95, S_current * 1.05 # Use more stable points
    # Make time points dynamic based on the data to avoid extrapolation
    T_short = df_options['TimeToExpiry'].min()
    T_long = df_options['TimeToExpiry'].max()

    # Find the nearest indices in the grids for our reference points
    s_low_idx = torch.argmin(torch.abs(s_range - S_low))
    s_high_idx = torch.argmin(torch.abs(s_range - S_high))
    s_atm_idx = torch.argmin(torch.abs(s_range - S_current))
    t_short_idx = torch.argmin(torch.abs(t_range - T_short)) # Use the shortest expiry
    t_long_idx = torch.argmin(torch.abs(t_range - T_long)) # Use the longest expiry

    # Extract volatility values at these points from the SCALED surface
    vol_at_low_S = scaled_vol_surface[s_low_idx, t_short_idx]
    vol_at_high_S = scaled_vol_surface[s_high_idx, t_short_idx]
    vol_at_short_T = scaled_vol_surface[s_atm_idx, t_short_idx]
    vol_at_long_T = scaled_vol_surface[s_atm_idx, t_long_idx]

    # Calculate Skew Intensity: (Vol at low strike - Vol at high strike)
    # A positive value indicates the classic volatility "smirk"
    skew_intensity = (vol_at_low_S - vol_at_high_S) * 100

    # Calculate Term Structure Tilt: (Vol at long expiry - Vol at short expiry)
    term_structure_tilt = (vol_at_long_T - vol_at_short_T) * 100

    # 4. Plot the scaled surface
    fig = go.Figure(data=[go.Surface(z=scaled_vol_surface.cpu().numpy(), 
                                     x=t_grid.cpu().numpy(), 
                                     y=s_grid.cpu().numpy())])
    fig.update_layout(scene=dict(
                          xaxis=dict(title='Time to Expiry (Years)', nticks=10),
                          yaxis=dict(title='Spot Price (S)'),
                          zaxis=dict(title='Scaled Local Volatility (σ)'),
                          camera=dict(eye=dict(x=1.6, y=1.6, z=1.6)) # Zoom in slightly
                      ),
                      autosize=False,
                      height=470, # Make plots shorter
                      margin=dict(l=65, r=50, b=65, t=40), # Reduced top margin
                      scene_dragmode='turntable') # Allow turntable rotation
    return fig, skew_intensity.item(), term_structure_tilt.item()

def calculate_implied_vol_surface(price_model, df_options, S_fixed, r_fixed, device, input_min, input_max):
    """
    Helper function to calculate the implied volatility surface from a trained price model.
    """

    # Create a grid for K and T from the data ranges
    k_min, k_max = df_options['Strike'].min(), df_options['Strike'].max()
    t_min, t_max = df_options['TimeToExpiry'].min(), df_options['TimeToExpiry'].max()
    k_range = torch.linspace(k_min, k_max, 50, device=device)
    t_range = torch.linspace(t_min, t_max, 50, device=device)
    k_grid, t_grid = torch.meshgrid(k_range, t_range, indexing='ij')

    iv_surface = np.zeros(k_grid.shape)

    print("Calculating Implied Volatility surface... (this may take a moment)")
    # Loop through each point on the grid to calculate IV
    for i in range(len(k_range)):
        for j in range(len(t_range)):
            K = k_range[i].item()
            T = t_range[j].item()
            # Re-integrate the IV calculation logic here
            S_tensor = torch.tensor([[S_fixed]], device=device, dtype=torch.float32)
            K_tensor = torch.tensor([[K]], device=device, dtype=torch.float32)
            T_tensor = torch.tensor([[T]], device=device, dtype=torch.float32)
            price_input_norm = normalize_inputs(S_tensor, K_tensor, T_tensor, input_min, input_max)
            with torch.no_grad():
                C_model = price_model(price_input_norm).item()
            def objective_func(sigma):
                q_fixed = 0.015
                return black_scholes_call_torch(S_tensor.squeeze(), K_tensor.squeeze(), T_tensor.squeeze(), torch.tensor(r_fixed, device=device), torch.tensor(sigma, device=device), torch.tensor(q_fixed, device=device)).item() - C_model
            try:
                iv_surface[i, j] = opt.brentq(objective_func, 0.01, 2.0)
            except (ValueError, RuntimeError):
                iv_surface[i, j] = np.nan

    iv_min = np.nanmin(iv_surface)
    iv_max = np.nanmax(iv_surface)

    return iv_surface, k_grid, t_grid, iv_min, iv_max

def plot_implied_vol_surface(iv_surface, k_grid, t_grid):
    """
    Plots the pre-calculated Implied Volatility (IV) surface.
    """

    # Plot the 3D surface
    fig = go.Figure(data=[go.Surface(z=iv_surface, x=t_grid.cpu().numpy(), y=k_grid.cpu().numpy())])
    fig.update_layout(scene=dict(
                          xaxis=dict(title='Time to Expiry (Years)', nticks=10),
                          yaxis=dict(title='Strike Price (K)'),
                          zaxis=dict(title='Implied Volatility (σ)'),
                          camera=dict(eye=dict(x=1.6, y=1.6, z=1.6)) # Zoom in slightly
                      ),
                      autosize=False, height=470, margin=dict(l=65, r=50, b=65, t=40), # Reduced top margin
                      scene_dragmode='turntable') # Allow turntable rotation
    return fig