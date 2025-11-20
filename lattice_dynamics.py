import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        self.t = 0.0
        
        # 1. Initialize Scalar Field (Smooth, not white noise)
        x = np.linspace(0, 2*np.pi, N)
        X, Y = np.meshgrid(x, x)
        self.S = np.exp(1j * (X + Y)) * 0.1 # Initial smooth state
        
        # 2. The "Firmware": Riemann Zeta Zeros
        # These define the harmonics of the vacuum.
        self.zeta_zeros = np.array([
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
            37.5861, 40.9187, 43.3270, 48.0051, 49.7738
        ])
        
        # 3. Spatial Mode Mapping (Geometry from Number Theory)
        # We map each Zeta zero to a specific spatial wave vector (kx, ky).
        # This creates a structured "Zeta Field" rather than static fuzz.
        self.wave_vectors = []
        rng = np.random.RandomState(42) # Fixed seed for reproducibility
        for z in self.zeta_zeros:
            angle = rng.uniform(0, 2*np.pi)
            # Scale wavenumber to fit grid (avoiding aliasing)
            k_mag = (z % 10) + 1 
            kx = k_mag * np.cos(angle)
            ky = k_mag * np.sin(angle)
            self.wave_vectors.append((kx, ky))
            
        # Pre-compute grid coordinates
        self.X, self.Y = X, Y

        # 4. Physics Parameters
        self.diffusion = 0.01    # Kinetic term strength
        self.mass = 0.05         # Mass term
        
        # Control Variable: The Coupling Temperature
        self.T = 0.005 
        
        # PID State
        self.integral_error = 0.0
        self.prev_error = 0.0

    def get_zeta_field(self):
        """
        Generates the vacuum field as a superposition of Zeta-waves.
        Phi(x,t) = Sum_n exp(i(k_n*x - w_n*t))
        """
        field = np.zeros((self.N, self.N), dtype=complex)
        
        for i, z in enumerate(self.zeta_zeros):
            kx, ky = self.wave_vectors[i]
            # Phase = k.x - omega.t
            phase = kx * self.X + ky * self.Y - z * self.t
            field += np.exp(1j * phase)
            
        # Normalize
        return field / np.sqrt(len(self.zeta_zeros))

    def measure_alpha(self):
        """
        Alpha = Interaction_Energy / Kinetic_Energy
        """
        # Kinetic Energy (Gradient)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2) + 1e-9
        
        # Interaction Energy (Coupling to Zeta Field)
        zeta_field = self.get_zeta_field()
        # Coherent interaction energy
        E_interaction = np.sum(np.abs(self.S * np.conj(zeta_field))) * self.T
        
        return E_interaction / E_kinetic

    def step_physics(self):
        self.t += 0.005 # Time step
        
        zeta_field = self.get_zeta_field()
        
        # Laplacian
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # Universal Update Rule
        # The Zeta field acts as a driving force (Pump)
        # The diffusion and mass act as restoring forces (Drain)
        
        # dS = i * (Kinetic + Potential + Source)
        update = 1j * (self.diffusion * laplacian - self.mass * self.S + self.T * zeta_field)
        
        self.S += update * 0.01
        
        # Soft Normalization (prevent explosion, allow breathing)
        norm = np.sqrt(np.sum(np.abs(self.S)**2))
        if norm > 0:
            self.S /= norm

    def apply_control(self, target_alpha):
        """
        Adjusts Temperature T to lock Alpha using Critical Damping.
        """
        current_alpha = self.measure_alpha()
        error = current_alpha - target_alpha
        
        # PID Coefficients for Critical Damping
        Kp = 0.01
        Ki = 0.0005
        Kd = 0.02 # ~ 2*sqrt(Kp)*zeta
        
        self.integral_error += error
        d_error = error - self.prev_error
        
        # Adjustment (Negative Feedback)
        # If Alpha is too high, we reduce T.
        adjustment = (Kp * error) + (Ki * self.integral_error) + (Kd * d_error)
        
        self.T -= adjustment * 0.1 # Scale down the correction step
        
        # Physical Bounds
        self.T = np.clip(self.T, 1e-7, 1.0)
        
        self.prev_error = error
        return current_alpha, self.T
