import torch

# ==============================================================================
#   LATTICE SPECTRAL DYNAMICS (LSD) CONFIGURATION
# ==============================================================================

# --- SYSTEM PARAMETERS ---
# Grid dimension N. Total cells = N^4.
# Recommended: 16 for consumer GPU, 64+ for HPC clusters.
GRID_SIZE = 16 

# Simulation device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.complex64

# --- PHYSICS CONSTANTS (NORMALIZED) ---
# Base oscillator frequency (Natural units: h_bar = c = 1)
OMEGA_0 = 1.0

# Spatial coupling constant (Diffusion rate / Kinetic term)
ALPHA_COUPLING = 0.1

# Self-interaction coupling (Analogous to gravitational backreaction)
KAPPA_GRAVITY = 0.05

# Noise coupling strength (Interaction with spectral reservoir)
EPSILON_NOISE = 0.01

# --- COSMOLOGICAL PARAMETERS ---
# System cooling rate per epoch.
# Represents dissipative losses or metric expansion.
# Critical value for stability is approx 0.92 for N=16.
COOLING_RATE = 0.92

# --- SPECTRAL NOISE SOURCE ---
# First 10 non-trivial zeros of the Riemann Zeta function (Im[rho])
# Used to generate GUE-compliant stochastic driving.
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010857, 30.424876, 32.935061, 
    37.586178, 40.918719, 43.327073, 48.005150, 49.773832
]
