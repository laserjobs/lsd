import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        self.t_step = 0.0
        
        # 1. Initialize Matter Field (The "Universe")
        self.S = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        self.S /= np.linalg.norm(self.S)
        
        # 2. RIEMANN ZETA DATA (The "Law")
        # The first 10 non-trivial zeros (Imaginary parts)
        # These define the resonant frequencies of the vacuum.
        self.zeta_zeros = np.array([
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
            37.5861, 40.9187, 43.3270, 48.0051, 49.7738
        ])
        
        # 3. Spatial Phase Map (Topology)
        # Each zero has a unique random phase structure across the grid.
        # This ensures the foam varies in space, not just time.
        self.zeta_phases = np.random.uniform(0, 2*np.pi, (len(self.zeta_zeros), N, N))

        # 4. Physics Constants
        self.omega_0 = 1.0
        self.temperature = 1.0e-5 # The variable we tune (Coupling Strength)
        
        # 5. Controller State
        self.prev_error = 0.0
        self.integral_error = 0.0

    def get_zeta_foam(self):
        """
        Generates 'Quantum Foam' based on Riemann Zeta Zeros.
        Instead of random noise, this is a structured interference pattern.
        """
        # Time evolution of the zeros: exp(i * gamma * t)
        # We sum over all zeros to get the local vacuum amplitude
        
        # Vectorized calculation:
        # We want: Sum_k [ exp(i * (gamma_k * t + phase_k(x,y))) ]
        
        # 1. Calculate time component (Shape: [10, 1, 1])
        time_phases = (self.zeta_zeros * self.t_step).reshape(-1, 1, 1)
        
        # 2. Add spatial phases (Shape: [10, 64, 64])
        total_phases = time_phases + self.zeta_phases
        
        # 3. Sum the waves
        foam_field = np.sum(np.exp(1j * total_phases), axis=0)
        
        # Normalize: Divide by sqrt(N_zeros) to keep energy consistent
        foam_field /= np.sqrt(len(self.zeta_zeros))
        
        return foam_field

    def measure_alpha(self):
        # Kinetic Energy (Matter Gradients)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2)
        
        # Interaction Energy (Matter coupling to Zeta Foam)
        # We recalculate the current foam state for the measurement
        foam = self.get_zeta_foam()
        
        # Interaction is weighted by the Temperature (Coupling Scale)
        E_interaction = np.sum(np.abs(self.S * foam * self.temperature))
        
        return E_interaction / (E_kinetic + 1e-9)

    def step_renormalization(self, target_alpha, zeta=0.707):
        # Increment Universe Time
        self.t_step += 0.01
        
        # --- A. PHYSICS ---
        # 1. Generate the Zeta Vacuum
        zeta_foam = self.get_zeta_foam()
        
        # 2. Calculate Energy Landscape
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        local_E = np.abs(grad_x)**2 + np.abs(grad_y)**2
        
        # 3. Dynamics
        omega_eff = self.omega_0 * (1 - 0.05 * local_E)
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # 4. Interaction Force
        # The matter field is driven by the Zeta Foam, scaled by Temperature
        driving_force = self.S * 1j * (zeta_foam * self.temperature)
        
        # 5. Update
        # Persistence 0.95 allows relaxation
        S_new = (self.S * 0.95) * np.exp(1j * omega_eff * 0.01) + \
                (0.01 * laplacian) + \
                (0.01 * driving_force)
                
        self.S = S_new / np.linalg.norm(S_new)

        # --- B. CONTROL (Tuning the Scale) ---
        current_alpha = self.measure_alpha()
        error = current_alpha - target_alpha
        
        # Tuned PID
        Kp = 0.02
        Ki = 0.0005
        Kd = zeta * 2 * np.sqrt(Kp)
        
        # Anti-Windup
        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -0.05, 0.05)
        
        d_error = error - self.prev_error
        adjustment = (Kp * error) + (Ki * self.integral_error) + (Kd * d_error)
        
        # Update Temperature (The Renormalization Scale)
        self.temperature -= adjustment * 5e-5
        self.temperature = np.clip(self.temperature, 1e-8, 1.0)
        
        self.prev_error = error
        
        return current_alpha, self.temperature
