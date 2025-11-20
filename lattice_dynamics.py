import torch
import config
from spectral_noise import SpectralNoiseGenerator

class LatticeDynamics:
    """
    Core physics engine for 4D Scalar Field Evolution.
    Implements Harmonic Stabilization (Ringing) logic.
    """
    def __init__(self, grid_size=config.GRID_SIZE, cooling_rate=config.COOLING_RATE):
        self.N = grid_size
        self.cooling_rate = cooling_rate
        self.device = config.DEVICE
        self.dt = 0.01
        self.time = 0.0
        
        # PID State for Harmonic Stabilization
        self.target_alpha = 1.0 / 137.035999
        self.integral_error = 0.0
        self.prev_error = 0.0 # For derivative term

        print(f"Initializing 4D Lattice (N={self.N}^4) on {self.device}...")
        
        # 1. Initialize State Field (psi)
        self.psi = torch.randn((self.N, self.N, self.N, self.N), 
                               dtype=config.DTYPE, device=self.device)
        self.normalize()
        
        # 2. Initialize Noise Generator
        self.noise_gen = SpectralNoiseGenerator(device=self.device)

    def normalize(self):
        norm = torch.sqrt(torch.sum(torch.abs(self.psi)**2))
        self.psi /= (norm + 1e-9)

    def compute_gradient_energy(self):
        grad_sq = torch.zeros_like(self.psi, dtype=torch.float32)
        for dim in range(4):
            diff = torch.roll(self.psi, shifts=1, dims=dim) - self.psi
            grad_sq += torch.abs(diff)**2
        return grad_sq

    def compute_laplacian(self):
        laplacian = -8.0 * self.psi
        for dim in range(4):
            laplacian += torch.roll(self.psi, shifts=1, dims=dim)
            laplacian += torch.roll(self.psi, shifts=-1, dims=dim)
        return laplacian

    def tune_cooling(self, current_alpha):
        """
        HARMONIC STABILIZATION:
        Adjusts cooling rate using second-order dynamics to induce
        damped oscillation (ringing) around the target Alpha.
        """
        error = current_alpha - self.target_alpha
        
        # Calculate derivative (rate of change of error)
        d_error = error - self.prev_error
        self.prev_error = error
        
        # PID Gains tuned for underdamped oscillation
        # High Kp = Strong restoring force (Oscillation)
        # Kd = Damping (Decay of oscillation)
        Kp = 5.0   
        Kd = 5.0   
        Ki = 0.1   
        
        self.integral_error += error
        
        # The "Force" on the cooling rate
        adjustment = (Kp * error) + (Kd * d_error) + (Ki * self.integral_error)
        
        # Apply adjustment
        # Factor 0.01 scales the PID output to the cooling_rate magnitude
        self.cooling_rate -= adjustment * 0.01
        
        # Clamp to physical bounds (Singularity vs. Freeze)
        self.cooling_rate = max(0.5, min(0.9999, self.cooling_rate))

    def measure_coupling_ratio(self):
        theta = self.noise_gen.get_field(self.time, self.psi.shape)
        e_int = torch.sum(torch.abs(self.psi) * torch.abs(theta)).item()
        e_geom = torch.sum(self.compute_gradient_energy()).item()
        if e_geom == 0: return 0.0
        return e_int / e_geom

    def step(self):
        E = self.compute_gradient_energy()
        omega_eff = config.OMEGA_0 * (1.0 - config.KAPPA_GRAVITY * E)
        phase_factor = torch.exp(1j * omega_eff * self.dt)
        diffusion = self.compute_laplacian() * config.ALPHA_COUPLING * self.dt
        theta = self.noise_gen.get_field(self.time, self.psi.shape)
        noise_kick = torch.exp(1j * theta)
        
        psi_next = (self.psi * phase_factor + diffusion) * noise_kick
        psi_next *= self.cooling_rate
        
        self.psi = psi_next
        self.normalize()
        self.time += self.dt
        
        # Enable the Harmonic Controller
        current_alpha = self.measure_coupling_ratio()
        self.tune_cooling(current_alpha)
