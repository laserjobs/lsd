import torch

# ==============================================================================
#   LATTICE SPECTRAL DYNAMICS (LSD) CONFIGURATION
#   Renormalized for Small-Grid (N=16) Convergence
# ==============================================================================

# --- SYSTEM PARAMETERS ---
# Grid dimension N. Total cells = N^4.
GRID_SIZE = 16 

# Simulation device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.complex64

# --- PHYSICS CONSTANTS (RENORMALIZED) ---
# Base oscillator frequency (Natural units: h_bar = c = 1)
OMEGA_0 = 1.0

# Spatial coupling constant (Diffusion rate / Kinetic term)
# INCREASED to 0.5 to stiffen the grid against low-N noise artifacts.
ALPHA_COUPLING = 0.5

# Self-interaction coupling (Analogous to gravitational backreaction)
KAPPA_GRAVITY = 0.05

# Noise coupling strength (Interaction with spectral reservoir)
# DECREASED to 0.0005 to prevent saturation in the small grid volume.
EPSILON_NOISE = 0.0005

# --- COSMOLOGICAL PARAMETERS ---
# System cooling rate per epoch.
# Starting value for the PID controller.
COOLING_RATE = 0.90

# --- SPECTRAL NOISE SOURCE ---
# First 10 non-trivial zeros of the Riemann Zeta function (Im[rho])
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010857, 30.424876, 32.935061, 
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832
]
