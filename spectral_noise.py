import torch
import numpy as np
import config

class SpectralNoiseGenerator:
    """
    Generates stochastic driving fields based on a defined spectral density.
    
    Implements the trace formula for quantum chaotic systems, where the
    driving noise theta(t) is synthesized from the eigenvalues of a 
    Random Matrix Ensemble (approximated here by Riemann Zeta zeros).
    """
    def __init__(self, device=config.DEVICE):
        self.device = device
        
        # Load spectral modes (Zeta zeros)
        self.modes = torch.tensor(config.ZETA_ZEROS, device=self.device, dtype=torch.float32)
        
        # Define spectral amplitudes
        # In GUE statistics, mode strength scales inversely with log(energy)
        self.amplitudes = 1.0 / torch.log(self.modes)
        
        # Cache for spatial phase map to ensure temporal coherence
        self.phase_map = None

    def get_field(self, time, grid_shape):
        """
        Computes the instantaneous noise field theta(x, t).
        
        Args:
            time (float): Current simulation time.
            grid_shape (tuple): Dimensions of the 4D lattice.
            
        Returns:
            torch.Tensor: The noise field tensor on the specified device.
        """
        # Initialize spatial phase map on first call
        # This represents the vacuum phase configuration
        if self.phase_map is None:
            self.phase_map = torch.rand(grid_shape, device=self.device) * 2 * np.pi

        # Compute the temporal component: Sum_n A_n * sin(omega_n * t)
        # Broadcasting allows efficient summation over all modes
        temporal_component = torch.sum(
            self.amplitudes[:, None] * torch.sin(self.modes[:, None] * time), 
            dim=0
        )
        
        # Combine temporal oscillation with spatial phase
        # The resulting field is non-local in momentum space but local in position space
        theta_field = torch.sin(temporal_component + self.phase_map)
        
        # Scale by the global coupling constant
        return theta_field * config.EPSILON_NOISE
