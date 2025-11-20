import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        self.t = 0.0
        
        # 1. Initialize Scalar Field
        x = np.linspace(0, 2*np.pi, N)
        X, Y = np.meshgrid(x, x)
        self.S = np.exp(1j * (X + Y)) 
        self.S /= np.linalg.norm(self.S)
        
        # 2. The "Firmware": Riemann Zeta Zeros
        self.zeta_zeros = np.array([
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
            37.5861, 40.9187, 43.3270, 48.0051, 49.7738
        ])
        
        # 3. Geometry Mapping
        self.wave_vectors = []
        rng = np.random.RandomState(42)
        for z in self.zeta_zeros:
            angle = rng.uniform(0, 2*np.pi)
            k_mag = (z % 5) + 1 
            kx = k_mag * np.cos(angle)
            ky = k_mag * np.sin(angle)
            self.wave_vectors.append((kx, ky))
            
        self.X, self.Y = X, Y

        # 4. Physics Parameters (CALIBRATED)
        # We increase these to "stiffen" the grid.
        # Higher diffusion = Higher Kinetic Energy = Lower Alpha for a given T.
        self.diffusion = 0.15   # Increased from 0.02
        self.mass = 0.2         # Increased from 0.1
        
        # Control Variable
        self.T = 0.01 # Start with a healthy energy
        
        # PID State
        self.integral_error = 0.0
        self.prev_error = 0.0

    def get_zeta_field(self):
        field = np.zeros((self.N, self.N), dtype=complex)
        for i, z in enumerate(self.zeta_zeros):
            kx, ky = self.wave_vectors[i]
            phase = kx * self.X + ky * self.Y - z * self.t
            field += np.exp(1j * phase)
        return field / np.sqrt(len(self.zeta_zeros))

    def measure_alpha(self):
        # Kinetic Energy (Stiffness)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2) + 1e-9
        
        # Interaction Energy (Zeta Noise)
        zeta_field = self.get_zeta_field()
        overlap = np.sum(self.S * np.conj(zeta_field))
        E_interaction = np.abs(overlap) * self.T 
        
        return E_interaction / E_kinetic

    def step_physics(self):
        self.t += 0.01
        zeta_field = self.get_zeta_field()
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # The Universal Update Rule
        update = 1j * (self.diffusion * laplacian - self.mass * self.S + self.T * zeta_field)
        self.S += update * 0.01
        self.S /= np.linalg.norm(self.S)

    def apply_control(self, target_alpha):
        current_alpha = self.measure_alpha()
        error = current_alpha - target_alpha
        
        # Tuned PID
        Kp = 0.1
        Ki = 0.005
        Kd = 0.2
        
        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -0.5, 0.5)
        d_error = error - self.prev_error
        
        adjustment = (Kp * error) + (Ki * self.integral_error) + (Kd * d_error)
        
        # Limit Slew Rate
        max_change = self.T * 0.05 + 1e-5
        adjustment = np.clip(adjustment, -max_change, max_change)
        
        self.T -= adjustment
        
        # Lower the floor slightly to allow breathing room
        self.T = np.clip(self.T, 1e-5, 1.0)
        
        self.prev_error = error
        return current_alpha, self.T
