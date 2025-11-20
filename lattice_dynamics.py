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

        print(f"Initializing 4D Lattice (N={self.N}^4) on {self.device}...")
        
        # 1. Initialize State Field (psi)
        # Complex scalar field representing the vacuum state
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
        """
        Calculates the local energy density E ~ |grad(psi)|^2.
        Used to compute the backreaction on the effective frequency.
        """
        grad_sq = torch.zeros_like(self.psi, dtype=torch.float32)
        # Finite difference over all 4 dimensions
        for dim in range(4):
            # Periodic boundary conditions (toroidal topology)
            diff = torch.roll(self.psi, shifts=1, dims=dim) - self.psi
            grad_sq += torch.abs(diff)**2
        return grad_sq

    def compute_laplacian(self):
        """
        Computes the discrete 4D Laplacian operator.
        Stencil: Sum(neighbors) - 8*center
        """
        laplacian = -8.0 * self.psi
        for dim in range(4):
            laplacian += torch.roll(self.psi, shifts=1, dims=dim)
            laplacian += torch.roll(self.psi, shifts=-1, dims=dim)
        return laplacian

    def step(self):
        """
        Performs one time-step of the evolution.
        """
        # A. Calculate Local Self-Interaction (Backreaction)
        # Gravity-like term: High energy density slows phase evolution
        E = self.compute_gradient_energy()
        omega_eff = config.OMEGA_0 * (1.0 - config.KAPPA_GRAVITY * E)
        
        # B. Compute Hamiltonian Dynamics
        # 1. Unitary Phase Rotation
        phase_factor = torch.exp(1j * omega_eff * self.dt)
        
        # 2. Kinetic/Diffusion Term
        diffusion = self.compute_laplacian() * config.ALPHA_COUPLING * self.dt
        
        # 3. Stochastic Driving (Spectral Noise)
        theta = self.noise_gen.get_field(self.time, self.psi.shape)
        noise_kick = torch.exp(1j * theta)
        
        # C. Update State
        # Combine: (Rotation + Diffusion) modulated by Noise
        psi_next = (self.psi * phase_factor + diffusion) * noise_kick
        
        # D. Dissipation (Cooling)
        # Simulates energy loss to an external bath or metric expansion
        psi_next *= self.cooling_rate
        
        # E. Finalize
        self.psi = psi_next
        self.normalize()
        self.time += self.dt

    def measure_coupling_ratio(self):
        """
        Calculates the dimensionless coupling ratio (Emergent Alpha).
        Ratio of Interaction Energy (Noise-Field coupling) to Geometric Energy.
        """
        # Re-calculate noise field for consistency
        theta = self.noise_gen.get_field(self.time, self.psi.shape)
        
        # Interaction Energy: < psi | theta | psi >
        e_int = torch.sum(torch.abs(self.psi) * torch.abs(theta)).item()
        
        # Geometric/Gradient Energy
        e_geom = torch.sum(self.compute_gradient_energy()).item()
        
        if e_geom == 0: return 0.0
        return e_int / e_geom
