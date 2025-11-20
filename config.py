import torch

# ==============================================================================
#   LATTICE SPECTRAL DYNAMICS (LSD) CONFIGURATION
#   Renormalized for Small-Grid (N=16) Convergence
# ==============================================================================

# --- SYSTEM PARAMETERS ---
# Grid dimension N. Total cells = N^4.
GRID_SIZE = 32 

# Simulation device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.complex64

# --- PHYSICS CONSTANTS (FINAL CALIBRATION) ---
# Base oscillator frequency
OMEGA_0 = 1.0

# Spatial coupling (MAXIMUM STIFFNESS)
# Forces rapid equilibration of energy across the small grid
ALPHA_COUPLING = 1.0 

# Self-interaction coupling 
KAPPA_GRAVITY = 0.05

# Noise coupling strength 
# MINIMAL NOISE INJECTION to prevent saturation
EPSILON_NOISE = 0.0001

# --- COSMOLOGICAL PARAMETERS ---
# Aggressive initial cooling to prevent early-epoch inflation
COOLING_RATE = 0.80

# --- SPECTRAL NOISE SOURCE ---
# First 10 non-trivial zeros of the Riemann Zeta function (Im[rho])
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010857, 30.424876, 32.935061, 
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832
]
