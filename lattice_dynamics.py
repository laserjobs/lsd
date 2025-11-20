import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        self.t_step = 0.0
        
        # 1. Initialize the Scalar Field (The State of the Universe)
        # Complex field initialized with random vacuum fluctuations
        self.S = np.random.normal(0, 0.1, (N, N)) + 1j * np.random.normal(0, 0.1, (N, N))
        
        # 2. The "Firmware": Riemann Zeta Zeros
        # These are the resonant frequencies of the vacuum noise.
        # Imaginary parts of the first 10 non-trivial zeros.
        self.zeta_zeros = np.array([
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
            37.5861, 40.9187, 43.3270, 48.0051, 49.7738
        ])
        
        # Spatial phase locks (Topology)
        self.zeta_phases = np.random.uniform(0, 2*np.pi, (len(self.zeta_zeros), N, N))

        # 3. Physics Parameters
        self.mass = 0.5          # Mass term (provides "stiffness" to the grid)
        self.diffusion = 0.1     # Spatial coupling strength (alpha_geometric)
        
        # 4. The Control Variable: Effective Temperature (Coupling Strength)
        # We start high to simulate a "hot" early universe
        self.T = 0.1 
        
        # 5. Control System State (for Critical Damping)
        self.alpha_history = []
        self.integral_error = 0.0
        self.prev_error = 0.0

    def get_zeta_noise(self):
        """
        Generates the 'Orpheus Signal': Noise structured by the Riemann Zeros.
        """
        self.t_step += 0.01
        
        # Summation of quantum oscillators driven by Zeta frequencies
        # Field(x, t) = Sum_k [ exp(i * (gamma_k * t + phase_k(x))) ]
        
        time_component = (self.zeta_zeros * self.t_step).reshape(-1, 1, 1)
        waves = np.exp(1j * (time_component + self.zeta_phases))
        
        # Coherent superposition
        noise_field = np.sum(waves, axis=0)
        
        # Normalize to unit energy density
        return noise_field / np.sqrt(len(self.zeta_zeros))

    def measure_alpha(self):
        """
        Calculates the emergent Fine Structure Constant.
        Alpha = Interaction_Energy / Kinetic_Energy
        """
        # Kinetic Energy (Gradient term)
        # Uses interactions between neighbors (Laplacian)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2) + 1e-9
        
        # Interaction Energy (Coupling to the Zeta Field)
        # This represents the energy injected by the vacuum noise
        zeta_field = self.get_zeta_noise()
        E_interaction = np.sum(np.abs(self.S * zeta_field)) * self.T
        
        # The emergent coupling constant
        return E_interaction / E_kinetic

    def step_physics(self):
        """
        Evolves the universe by one time step.
        """
        # 1. Generate the Vacuum Geometry (Zeta Noise)
        noise = self.get_zeta_noise()
        
        # 2. Calculate Laplacian (Diffusion/Spatial Coupling)
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # 3. The Universal Update Rule (Discrete)
        # dS/dt = i * [ Kinetic + Mass + Interaction ]
        # Interaction strength is modulated by Temperature T
        dS = 1j * (self.diffusion * laplacian - self.mass * self.S + self.T * noise * self.S)
        
        # 4. Update State
        self.S += dS * 0.01 # dt = 0.01
        
        # Renormalize to maintain unitarity (Conservation of Probability)
        norm = np.sqrt(np.sum(np.abs(self.S)**2))
        self.S /= norm

    def apply_critical_control(self, target_alpha, zeta=0.707):
        """
        Adjusts the Vacuum Temperature T to stabilize Alpha at the target.
        Uses a PID controller tuned for Critical Damping.
        """
        current_alpha = self.measure_alpha()
        
        # Error: Distance from the physical constant
        error = current_alpha - target_alpha
        
        # --- Control Physics ---
        # We model the adjustment of T as a damped harmonic oscillator.
        # stiffness (Kp) determines how fast we correct.
        # damping (Kd) determines stability.
        
        Kp = 0.05  # Stiffness
        Ki = 0.002 # Integral (removes steady-state drift)
        
        # Critical Damping Relation: Kd = 2 * zeta * sqrt(Kp)
        Kd = 2 * zeta * np.sqrt(Kp) 
        
        # PID Terms
        P = Kp * error
        I = Ki * self.integral_error
        D = Kd * (error - self.prev_error)
        
        # The "Restoring Force" on the Temperature
        adjustment = P + I + D
        
        # Apply Adjustment (Negative feedback: if Alpha is too high, lower T)
        self.T -= adjustment
        
        # Constraints
        self.T = np.clip(self.T, 1e-6, 1.0) # Temperature must be positive
        self.integral_error += error
        self.prev_error = error
        
        return current_alpha, self.T
