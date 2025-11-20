import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        self.t = 0.0
        
        # 1. Initialize Scalar Field (Smooth Plane Wave)
        x = np.linspace(0, 2*np.pi, N)
        X, Y = np.meshgrid(x, x)
        # Initial state: A low-momentum plane wave
        self.S = np.exp(1j * (X + Y)) 
        self.S /= np.linalg.norm(self.S) # Normalize
        
        # 2. The "Firmware": Riemann Zeta Zeros
        self.zeta_zeros = np.array([
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
            37.5861, 40.9187, 43.3270, 48.0051, 49.7738
        ])
        
        # 3. Geometry Mapping
        # Map zeros to wave vectors to create a structured vacuum
        self.wave_vectors = []
        rng = np.random.RandomState(42)
        for z in self.zeta_zeros:
            angle = rng.uniform(0, 2*np.pi)
            k_mag = (z % 5) + 1 # Constrain wavenumbers to avoid aliasing on 64x64
            kx = k_mag * np.cos(angle)
            ky = k_mag * np.sin(angle)
            self.wave_vectors.append((kx, ky))
            
        self.X, self.Y = X, Y

        # 4. Physics Parameters
        self.diffusion = 0.02    
        self.mass = 0.1         
        
        # Control Variable: Coupling Temperature
        # SOFT START: Initialize close to the target (0.007) to prevent shock
        self.T = 0.0075 
        
        # PID State
        self.integral_error = 0.0
        self.prev_error = 0.0

    def get_zeta_field(self):
        """
        Constructs the vacuum field from Zeta modes.
        """
        field = np.zeros((self.N, self.N), dtype=complex)
        for i, z in enumerate(self.zeta_zeros):
            kx, ky = self.wave_vectors[i]
            # Propagating waves: exp(i(k.x - w.t))
            phase = kx * self.X + ky * self.Y - z * self.t
            field += np.exp(1j * phase)
        
        # Normalize energy density to 1
        return field / np.sqrt(len(self.zeta_zeros))

    def measure_alpha(self):
        """
        Alpha = Interaction / Kinetic
        """
        # 1. Kinetic Energy (Laplacian/Gradient)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2) + 1e-9
        
        # 2. Interaction Energy (Coupling to Zeta)
        zeta_field = self.get_zeta_field()
        # Coherent overlap integral
        overlap = np.sum(self.S * np.conj(zeta_field))
        E_interaction = np.abs(overlap) * self.T 
        
        return E_interaction / E_kinetic

    def step_physics(self):
        self.t += 0.01 # Time step
        
        zeta_field = self.get_zeta_field()
        
        # Spatial coupling
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # Evolution Equation: Schrödinger-like
        # dS/dt = i * ( H_kinetic + H_mass + H_interaction )
        update = 1j * (self.diffusion * laplacian - self.mass * self.S + self.T * zeta_field)
        
        # Apply update
        self.S += update * 0.01
        
        # Renormalize (Unitary evolution constraint)
        self.S /= np.linalg.norm(self.S)

    def apply_control(self, target_alpha):
        current_alpha = self.measure_alpha()
        error = current_alpha - target_alpha
        
        # Tuned PID for Stability
        Kp = 0.05
        Ki = 0.001
        Kd = 0.1
        
        self.integral_error += error
        # Anti-windup
        self.integral_error = np.clip(self.integral_error, -0.1, 0.1)
        
        d_error = error - self.prev_error
        
        adjustment = (Kp * error) + (Ki * self.integral_error) + (Kd * d_error)
        
        # SLEW RATE LIMIT: Prevent drastic changes
        # Max change per step is 1% of current T
        max_change = self.T * 0.01 + 1e-5
        adjustment = np.clip(adjustment, -max_change, max_change)
        
        # Negative Feedback: High Alpha -> Reduce T
        self.T -= adjustment
        
        # VACUUM FLOOR: Prevent T from reaching 0
        # We assume a minimum vacuum energy exists (Higgs VEV analogy)
        self.T = np.clip(self.T, 0.0001, 1.0)
        
        self.prev_error = error
        return current_alpha, self.T
