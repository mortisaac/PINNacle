import torch
import torch.nn as nn

from utils import normalize_inputs, generate_collocation_points

def calculate_pde_residual(price_model, vol_model, S, K, T, r, input_min, input_max):
    """
    Calculates the Black-Scholes PDE residual for a given model and inputs.

    Args:
        price_model (nn.Module): The neural network model that predicts option price C.
        vol_model (nn.Module): The neural network model that predicts local volatility sigma_LV.
        S (torch.Tensor): Spot price tensor (un-normalized).
        K (torch.Tensor): Strike price tensor (un-normalized).
        T (torch.Tensor): Time to maturity tensor (un-normalized).
        r (torch.Tensor): Risk-free interest rate.
        input_min (torch.Tensor): Tensor of minimum values for [S, K, T] for normalization.
        input_max (torch.Tensor): Tensor of maximum values for [S, K, T] for normalization.

    Returns:
        tuple: A tuple containing (pde_residual, sigma_LV).
    """
    # Ensure S and T are tracking gradients for derivative calculation
    S.requires_grad_()
    T.requires_grad_()

    # Reshape inputs for the model
    S_r = S.unsqueeze(1)
    K_r = K.unsqueeze(1)
    T_r = T.unsqueeze(1)

    # Normalize inputs before passing them to the models
    price_inputs_norm = normalize_inputs(S_r, K_r, T_r, input_min, input_max)
    vol_inputs_norm = price_inputs_norm[:, [0, 2]] # Select normalized S and T

    # 1. Get the price C_hat from the price_model using normalized inputs
    C_hat = price_model(price_inputs_norm)

    # 2. Get the local volatility sigma_LV from the vol_model
    sigma_LV = vol_model(vol_inputs_norm)

    # 3. Calculate first-order derivatives (dC/dT, dC/dS) from C_hat
    dC_dT_r, dC_dS_r = torch.autograd.grad(C_hat, [T_r, S_r], grad_outputs=torch.ones_like(C_hat), create_graph=True)

    # 4. Calculate second-order derivative (d2C/dS2 or Gamma)
    d2C_dS2_r = torch.autograd.grad(dC_dS_r, S_r, grad_outputs=torch.ones_like(dC_dS_r), create_graph=True)[0]

    # 5. Calculate the Black-Scholes PDE residual using the derivatives and learned sigma_LV
    # The PDE is defined in terms of t (time from now), but our input is T (time to expiry).
    # Therefore, dC/dt = -dC/dT. We must flip the sign of the calculated dC_dT.
    q = 0.015 # Assumed annualized dividend yield for SPY
    residual = -dC_dT_r.squeeze() + (r - q) * S * dC_dS_r.squeeze() + 0.5 * sigma_LV.squeeze()**2 * S**2 * d2C_dS2_r.squeeze() - r * C_hat.squeeze()
    # Scale the residual by the spot price to make the loss term more stable and less
    # sensitive to the absolute magnitude of the derivatives.
    return residual / (S + 1e-8), sigma_LV

def pinn_loss_function(price_model, vol_model, df_options, S_fixed, r_fixed, collocation_points, device, weights, input_min, input_max):
    """
    Calculates the combined loss for the dual-model PINN.

    This includes:
    1. Data Loss: MSE between predicted price and market price.
    2. PDE Loss: Residual of the Black-Scholes PDE.
    3. Boundary Loss: Enforces the terminal condition (payoff at expiry).

    Args:
        price_model (nn.Module): The network that predicts option price C(S, K, T).
        vol_model (nn.Module): The network that predicts local volatility sigma(S, T).
        df_options (pd.DataFrame): DataFrame with market option data (Strike, TimeToExpiry, MarketPrice).
        S_fixed (float): The fixed spot price of the underlying asset.
        r_fixed (float): The fixed risk-free interest rate.
        collocation_points (torch.Tensor): Randomly sampled points [S', K', T'] for PDE loss.
        device (torch.device): The device (CPU or GPU) to perform calculations on.
        weights (dict): A dictionary with weights for 'data', 'pde', and 'boundary' losses.
        input_min (torch.Tensor): Tensor of minimum values for [S, K, T] for normalization.
        input_max (torch.Tensor): Tensor of maximum values for [S, K, T] for normalization.

    Returns:
        tuple: A tuple containing (total_loss, loss_data, loss_pde, loss_boundary, avg_sigma).
    """

    # --- 1. Data Loss (L_data) ---
    # Prepare tensors from the full market option data (calls and puts)
    K_data = torch.tensor(df_options['Strike'].values, dtype=torch.float32).unsqueeze(1).to(device)
    T_data = torch.tensor(df_options['TimeToExpiry'].values, dtype=torch.float32).unsqueeze(1).to(device)
    S_data = torch.full_like(K_data, S_fixed, device=device)
    market_price = torch.tensor(df_options['MarketPrice'].values, dtype=torch.float32).unsqueeze(1).to(device)
    # Create a boolean tensor to identify call options
    is_call_data = torch.tensor(df_options['Type'].values == 'call', dtype=torch.bool).unsqueeze(1).to(device)

    # Normalize the data inputs
    price_inputs_data_norm = normalize_inputs(S_data, K_data, T_data, input_min, input_max)

    # Get predicted price C_hat from the price_model
    C_hat_data = price_model(price_inputs_data_norm)
    
    # Use Relative Mean Squared Error to handle the wide range of option prices.
    # This prevents high-priced options from dominating the loss.
    loss_data = torch.mean(((C_hat_data - market_price) / (market_price + 1e-8))**2)

    # --- 2. Physics Loss (L_pde) ---
    # Unpack collocation points
    S_col, K_col, T_col = collocation_points.T.to(device)
    r_col = torch.full_like(S_col, r_fixed).to(device)
    
    # We must pass the original, un-normalized tensors to calculate_pde_residual.
    # The normalization will happen inside that function before calling the models.
    pde_residual, sigma_lv_col = calculate_pde_residual(price_model, vol_model, S_col, K_col, T_col, r_col, input_min, input_max)
    # The residual should be zero, so we compute MSE against a zero tensor
    loss_pde = nn.functional.mse_loss(pde_residual, torch.zeros_like(pde_residual).to(device))

    # Calculate the average of the sampled local volatilities
    avg_sigma = torch.mean(sigma_lv_col)

    # --- 3. Boundary Loss (L_boundary) ---
    # Enforce the terminal condition C(S, K, T=0) = max(S - K, 0)
    # We use collocation points for S and K, but fix T to zero.
    T_boundary = torch.zeros_like(S_col) # Terminal condition is at T=0

    # Normalize the boundary inputs
    boundary_inputs_norm = normalize_inputs(S_col.unsqueeze(1), K_col.unsqueeze(1), T_boundary.unsqueeze(1), input_min, input_max)

    # Get the model's price prediction at the terminal time
    C_hat_boundary = price_model(boundary_inputs_norm)

    # Calculate the known payoff at expiry for both calls and puts
    # We can randomly assign option types to the collocation points for the boundary condition
    is_call_boundary = (torch.rand_like(S_col) > 0.5).unsqueeze(1)
    call_payoff = torch.relu(S_col.unsqueeze(1) - K_col.unsqueeze(1))
    put_payoff = torch.relu(K_col.unsqueeze(1) - S_col.unsqueeze(1))
    payoff = torch.where(is_call_boundary, call_payoff, put_payoff)

    # Use a scaled error for the boundary loss to make it less sensitive to the
    # absolute magnitude of the payoff, which can be very large for ITM options.
    loss_boundary = torch.mean(((C_hat_boundary - payoff) / S_col.unsqueeze(1))**2)

    # --- 4. Combined Loss ---
    total_loss = (weights['data'] * loss_data +
                  weights['pde'] * loss_pde +
                  weights['boundary'] * loss_boundary)

    return total_loss, loss_data, loss_pde, loss_boundary, avg_sigma

def train_pinn(price_model, vol_model, df_options, S_fixed, r_fixed, device, epochs=5000, pretrain_epochs=2000, lr=0.001, input_min=None, input_max=None):
    """
    Trains the dual-model Physics-Informed Neural Network.

    Args:
        price_model (nn.Module): The network for option price C(S, K, T).
        vol_model (nn.Module): The network for local volatility sigma(S, T).
        df_options (pd.DataFrame): DataFrame with market option data.
        S_fixed (float): The fixed spot price.
        r_fixed (float): The fixed risk-free interest rate.
        device (torch.device): The device to train on (CPU or GPU).
        epochs (int): Number of training epochs.
        pretrain_epochs (int): Number of epochs for data-only pre-training.
        lr (float): Learning rate for the optimizer.
        input_min (torch.Tensor): Pre-calculated min values for normalization.
        input_max (torch.Tensor): Pre-calculated max values for normalization.
    """

    # --- Stage 1: Pre-training on Market Data ---
    print(f"\n--- Starting Stage 1: Pre-training for {pretrain_epochs} epochs (data-only) ---")
    # Create a dedicated optimizer and scheduler for the pre-training stage.
    pretrain_optimizer = torch.optim.Adam(price_model.parameters(), lr=lr)
    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=pretrain_epochs)

    pretrain_loss_weights = {'data': 100.0, 'pde': 0.0, 'boundary': 0.0}
    # Temporarily disable gradients for the volatility network during pre-training.
    for param in vol_model.parameters():
        param.requires_grad = False
    for epoch in range(pretrain_epochs):
        # In this stage, we only need to fit the market data, so collocation points are not strictly necessary
        # for the loss calculation. We calculate the data loss directly here for maximum efficiency.
        
        # --- Simplified Data Loss for Pre-training ---
        K_data = torch.tensor(df_options['Strike'].values, dtype=torch.float32).unsqueeze(1).to(device)
        T_data = torch.tensor(df_options['TimeToExpiry'].values, dtype=torch.float32).unsqueeze(1).to(device)
        S_data = torch.full_like(K_data, S_fixed, device=device)
        market_price = torch.tensor(df_options['MarketPrice'].values, dtype=torch.float32).unsqueeze(1).to(device)

        price_inputs_data_norm = normalize_inputs(S_data, K_data, T_data, input_min, input_max)
        C_hat_data = price_model(price_inputs_data_norm)
        
        # Use Relative MSE for pre-training loss
        loss_data = torch.mean(((C_hat_data - market_price) / (market_price + 1e-8))**2)
        
        # The total loss for this stage is just the weighted data loss.
        loss = pretrain_loss_weights['data'] * loss_data
        # --- End of Simplified Data Loss ---

        pretrain_optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients from the relative error term,
        # which can be unstable for low-priced options.
        torch.nn.utils.clip_grad_norm_(price_model.parameters(), max_norm=1.0)

        pretrain_optimizer.step()
        pretrain_scheduler.step()

        if epoch % 100 == 0:
            print(f"Pre-train Epoch {epoch}/{pretrain_epochs} | Loss: {loss.item():.6f} | LR: {pretrain_scheduler.get_last_lr()[0]:.2e}")

    # Re-enable gradients for the volatility network for the main training stage.
    for param in vol_model.parameters():
        param.requires_grad = True

    # --- Stage 2: Full PINN Training (Data + Physics) ---
    print(f"\n--- Starting Stage 2: Full PINN Training for {epochs} epochs ---")
    optimizer = torch.optim.Adam(
        list(price_model.parameters()) + list(vol_model.parameters()), 
        lr=lr/10
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Dynamically scale the PDE loss weight based on the stock price.
    # The PDE term has S^2, which can cause the loss to explode for high-priced stocks.
    # We use SPY's approx price (~500) as a baseline.
    price_scale_factor = (500 / S_fixed) ** 2 if S_fixed > 0 else 1.0
    pde_weight_final = 5e-3 * price_scale_factor
    print(f"Using price-scaled final PDE weight: {pde_weight_final:.2e}")

    pinn_loss_weights = {'data': 100.0, 'pde': 1e-7, 'boundary': 1.0}

    for epoch in range(epochs):
        pinn_loss_weights['pde'] = 1e-7 + (pde_weight_final - 1e-7) * (epoch / epochs)

        # Generate new random collocation points for each epoch
        collocation_points = generate_collocation_points(df_options, S_fixed, n_points=10000)

        loss, loss_data, loss_pde, loss_boundary, avg_sigma = pinn_loss_function(
            price_model, vol_model, df_options, S_fixed, r_fixed, collocation_points, device, pinn_loss_weights, input_min, input_max
        )

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients for the main training stage as well.
        torch.nn.utils.clip_grad_norm_(price_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(vol_model.parameters(), max_norm=1.0)

        # Step the optimizer with the combined gradients
        optimizer.step()

        # Step the scheduler at the end of each epoch
        scheduler.step()

        if epoch % 100 == 0:
            print(f"PINN Epoch {epoch}/{epochs} | Total Loss: {loss.item():.6f} | "
                  f"Data: {loss_data.item():.6f} | PDE: {loss_pde.item():.6f} | Boundary: {loss_boundary.item():.6f} | Avg Vol: {avg_sigma.item():.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | PDE Weight: {pinn_loss_weights['pde']:.2e}")

    print("Training finished.")