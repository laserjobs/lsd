import torch

# ==============================================================================
#   LATTICE SPECTRAL DYNAMICS (LSD) CONFIGURATION
#   Renormalized for Small-Grid (N=16) Convergence
# ==============================================================================

GRID_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.complex64

# Physics
OMEGA_0 = 1.0
ALPHA_COUPLING = 0.1  # Reset to standard
KAPPA_GRAVITY = 0.05
EPSILON_NOISE = 0.01  # Reset to standard

# Cosmology
COOLING_RATE = 0.95   # Start neutral, let PID find the value

# --- SPECTRAL NOISE SOURCE ---
# First 10 non-trivial zeros of the Riemann Zeta function (Im[rho])
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010857, 30.424876, 32.935061, 
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832
]
