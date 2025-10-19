import torch.nn as nn

class PriceNetwork(nn.Module):
    """
    Neural Network to approximate the option price directly.
    Outputs a non-negative price using a Softplus activation.
    """
    def __init__(self):
        super(PriceNetwork, self).__init__()
        
        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(3, 200),   # Input layer: S, K, T
            nn.Tanh(),
            nn.Linear(200, 200),  # Hidden layer 1
            nn.Tanh(),
            nn.Linear(200, 200),  # Hidden layer 2
            nn.Tanh(),
            nn.Linear(200, 200),  # Hidden layer 3
            nn.Tanh(),
            nn.Linear(200, 200),  # Hidden layer 4
            nn.Tanh(),
            nn.Linear(200, 200),  # Hidden layer 5
            nn.Tanh(),
            nn.Linear(200, 200),  # Hidden layer 6
            nn.Tanh(),
            nn.Linear(200, 1),   # Output layer: Option Price
            nn.Softplus()       # Ensure non-negative output for price
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.net(x)

class VolatilityNetwork(nn.Module):
    """
    Neural Network to approximate local volatility.
    Outputs volatility constrained to a reasonable range [0.01, 0.81].
    """
    def __init__(self):
        super(VolatilityNetwork, self).__init__()
        
        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(2, 50),   # Input layer: S, T
            nn.Tanh(),
            nn.Linear(50, 50),  # Hidden layer 1
            nn.Tanh(),
            nn.Linear(50, 50),  # Hidden layer 2
            nn.Tanh(),
            nn.Linear(50, 50),  # Hidden layer 3
            nn.Tanh(),
            nn.Linear(50, 50),  # Hidden layer 4
            nn.Tanh(),
            nn.Linear(50, 1),   # Output layer: Local Volatility (unconstrained)
            nn.Softplus()       # Use Softplus to ensure non-negative volatility without a hard floor/ceiling.
        )

    def forward(self, x):
        """
        Forward pass with output scaling.
        """
        return self.net(x)