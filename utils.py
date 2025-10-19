import torch
import numpy as np
import pandas as pd

def generate_collocation_points(df_options, S_fixed, n_points=10000):
    """
    Generates random collocation points by sampling from the provided data's distributions.
    """
    n_data_points = len(df_options)
    sample_indices = np.random.choice(n_data_points, n_points, replace=True)
    S_col = torch.full((n_points, 1), S_fixed)
    K_col = torch.tensor(df_options['Strike'].iloc[sample_indices].values, dtype=torch.float32).unsqueeze(1)
    T_col = torch.tensor(df_options['TimeToExpiry'].iloc[sample_indices].values, dtype=torch.float32).unsqueeze(1)
    return torch.cat([S_col, K_col, T_col], dim=1)

def get_normalization_stats(df, S_fixed):
    """
    Calculates the min and max values for input features (S, K, T) for normalization.
    """
    s_min, s_max = S_fixed * 0.8, S_fixed * 1.2
    k_min, k_max = df['Strike'].min(), df['Strike'].max()
    t_min, t_max = df['TimeToExpiry'].min(), df['TimeToExpiry'].max()
    input_min = torch.tensor([s_min, k_min, t_min], dtype=torch.float32)
    input_max = torch.tensor([s_max, k_max, t_max], dtype=torch.float32)
    return input_min, input_max

def normalize_inputs(S, K, T, input_min, input_max):
    """
    Normalizes S, K, and T tensors to the [0, 1] range.
    """
    X = torch.cat([S, K, T], dim=1)
    X_normalized = (X - input_min) / (input_max - input_min + 1e-8)
    X_clamped = torch.clamp(X_normalized, 0, 1)
    return X_clamped

def black_scholes_call_torch(S, K, T, r, sigma, q):
    """
    Calculates the Black-Scholes-Merton price for a European call option using torch tensors.
    This internal implementation avoids dependency issues with external libraries.
    """
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T) + 1e-8)
    d2 = d1 - sigma * torch.sqrt(T)
    normal_dist = torch.distributions.Normal(0, 1)
    call_price = S * torch.exp(-q * T) * normal_dist.cdf(d1) - K * torch.exp(-r * T) * normal_dist.cdf(d2)
    return call_price
def calculate_greeks(price_model, S, K, T, input_min, input_max, get_gamma=False):
    """
    Calculates the Option Greeks (Delta, Gamma, Theta) for a given set of inputs.

    Args:
        price_model (nn.Module): The trained neural network for option price.
        S (torch.Tensor): 1D Spot price tensor.
        K (torch.Tensor): 1D Strike price tensor.
        T (torch.Tensor): 1D Time to maturity tensor.
        input_min (torch.Tensor): Normalization minimums.
        input_max (torch.Tensor): Normalization maximums.
        get_gamma (bool): If True, calculates Gamma (slower).

    Returns:
        tuple: A tuple containing (price, delta, gamma, theta_per_day).
    """
    # Ensure inputs require gradients for autograd
    S.requires_grad_()
    T.requires_grad_()

    # Reshape inputs to match the shape used during training's PDE loss calculation.
    # This consistency is critical for autograd to function correctly.
    S_r = S.unsqueeze(1)
    K_r = K.unsqueeze(1)
    T_r = T.unsqueeze(1)
    price_inputs_norm = normalize_inputs(S_r, K_r, T_r, input_min, input_max)

    # Get the price C_hat from the model
    C_hat = price_model(price_inputs_norm)

    # Calculate first-order derivatives. The gradient must be taken with respect to
    # the tensors that were directly used to compute C_hat (i.e., S_r and T_r).
    dC_dT_r, dC_dS_r = torch.autograd.grad(C_hat, [T_r, S_r], grad_outputs=torch.ones_like(C_hat), create_graph=get_gamma)

    delta = dC_dS_r.squeeze()
    theta_per_day = -dC_dT_r.squeeze() / 365.25
    
    gamma = torch.tensor(0.0) # Default gamma to 0
    if get_gamma:
        # Calculate second-order derivative (d2C/dS2 or Gamma)
        d2C_dS2_r = torch.autograd.grad(dC_dS_r, S_r, grad_outputs=torch.ones_like(dC_dS_r), create_graph=False)[0]
        gamma = d2C_dS2_r.squeeze()

    return C_hat.item(), delta.item(), gamma.item(), theta_per_day.item()