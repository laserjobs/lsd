import torch
import config
from spectral_noise import SpectralNoiseGenerator

class LatticeDynamics:
    """
    Core physics engine for 4D Scalar Field Evolution.
    
    Solves the discrete equation of motion:
    d_psi/dt = -i * H_eff * psi + Dissipation + Noise
    """
    def __init__(self, grid_size=config.GRID_SIZE, cooling_rate=config.COOLING_RATE):
        self.N = grid_size
        self.cooling_rate = cooling_rate
        self.device = config.DEVICE
        self.dt = 0.01
        self.time = 0.0
        
        # Tuning Parameters for PID Controller
        self.target_alpha = 1.0 / 137.035999
        self.integral_error = 0.0

        print(f"Initializing 4D Lattice (N={self.N}^4) on {self.device}...")
        
        # 1. Initialize State Field (psi)
        self.psi = torch.randn((self.N, self.N, self.N, self.N), 
                               dtype=config.DTYPE, device=self.device)
        self.normalize()
        
        # 2. Initialize Noise Generator
        self.noise_gen = SpectralNoiseGenerator(device=self.device)

    def normalize(self):
        """Enforces global unitarity (Conservation of Probability)."""
        norm = torch.sqrt(torch.sum(torch.abs(self.psi)**2))
        self.psi /= (norm + 1e-9)

    def compute_gradient_energy(self):
        """Calculates local energy density E ~ |grad(psi)|^2."""
        grad_sq = torch.zeros_like(self.psi, dtype=torch.float32)
        for dim in range(4):
            diff = torch.roll(self.psi, shifts=1, dims=dim) - self.psi
            grad_sq += torch.abs(diff)**2
        return grad_sq

    def compute_laplacian(self):
        """Computes the discrete 4D Laplacian operator."""
        laplacian = -8.0 * self.psi
        for dim in range(4):
            laplacian += torch.roll(self.psi, shifts=1, dims=dim)
            laplacian += torch.roll(self.psi, shifts=-1, dims=dim)
        return laplacian

    def tune_cooling(self, current_alpha):
        """
        AUTO-CALIBRATION: Adjusts the cooling rate to stabilize Alpha.
        Implements a PI (Proportional-Integral) Controller.
        """
        error = current_alpha - self.target_alpha
        self.integral_error += error
        
        # Controller Gains (Tuned for N=16 stability)
        Kp = 0.1
        Ki = 0.01
        
        # Adjustment: If Alpha is too high, we need MORE cooling (lower rate)
        adjustment = (Kp * error) + (Ki * self.integral_error)
        
        # Apply adjustment (inverted because lower rate = more cooling)
        self.cooling_rate -= adjustment
        
        # Clamp to physical bounds to prevent collapse or explosion
        self.cooling_rate = max(0.85, min(0.999, self.cooling_rate))

    def measure_coupling_ratio(self):
        """Calculates the emergent Alpha."""
        theta = self.noise_gen.get_field(self.time, self.psi.shape)
        e_int = torch.sum(torch.abs(self.psi) * torch.abs(theta)).item()
        e_geom = torch.sum(self.compute_gradient_energy()).item()
        if e_geom == 0: return 0.0
        return e_int / e_geom

    def step(self):
        """Performs one time-step of the evolution."""
        # A. Calculate Local Self-Interaction
        E = self.compute_gradient_energy()
        omega_eff = config.OMEGA_0 * (1.0 - config.KAPPA_GRAVITY * E)
        
        # B. Compute Hamiltonian Dynamics
        phase_factor = torch.exp(1j * omega_eff * self.dt)
        diffusion = self.compute_laplacian() * config.ALPHA_COUPLING * self.dt
        theta = self.noise_gen.get_field(self.time, self.psi.shape)
        noise_kick = torch.exp(1j * theta)
        
        # C. Update State
        psi_next = (self.psi * phase_factor + diffusion) * noise_kick
        
        # D. Dissipation (Dynamic Cooling)
        psi_next *= self.cooling_rate
        
        # E. Finalize
        self.psi = psi_next
        self.normalize()
        self.time += self.dt
        
        # F. Auto-Tune (Feedback Loop)
        # Check alpha every step for tight control loop
        current_alpha = self.measure_coupling_ratio()
        self.tune_cooling(current_alpha)
