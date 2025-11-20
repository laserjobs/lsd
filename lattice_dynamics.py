import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        self.t_step = 0.0
        
        # 1. Initialize Complex Scalar Field
        self.S = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        self.S /= np.linalg.norm(self.S)
        
        # 2. RIEMANN ZETA ZEROS (The DNA of the Vacuum)
        # These imaginary parts define the "Resonant Frequencies" of the quantum foam.
        self.zeta_zeros = np.array([
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
            37.5861, 40.9187, 43.3270, 48.0051, 49.7738,
            52.9703, 56.4462, 59.3470, 60.8317, 65.1125
        ])
        
        # Assign random spatial phase maps to each zero (Topology)
        # This creates a complex 2D interference pattern for each frequency
        self.zeta_phases = np.random.uniform(0, 2*np.pi, (len(self.zeta_zeros), N, N))

        # 3. Physics Constants
        self.omega_0 = 1.0
        self.mass = 0.1          # Mass term to stabilize the vacuum (Higgs-like)
        self.temperature = 0.001 # Starting Energy Scale
        
        # Vacuum Floor (The "Cosmological Constant" correction)
        # Prevents division by zero in Alpha calculation
        self.vacuum_floor = 1e-6

    def get_zeta_foam(self):
        """
        Generates the background metric noise based on Riemann Zeros.
        The vacuum is not random; it is a symphony of prime-number resonances.
        """
        # Evolve time
        self.t_step += 0.01
        
        # Calculate the interference pattern at this time step
        # Sum( exp(i * (gamma*t + phase)) )
        time_phases = (self.zeta_zeros * self.t_step).reshape(-1, 1, 1)
        total_phases = time_phases + self.zeta_phases
        
        foam = np.sum(np.exp(1j * total_phases), axis=0)
        
        # Normalize standard deviation to ~1.0
        foam /= np.sqrt(len(self.zeta_zeros))
        return foam

    def measure_alpha(self):
        # 1. Kinetic Energy (Gradient term)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2)
        
        # 2. Interaction Energy (Coupling to Zeta Foam)
        zeta_foam = self.get_zeta_foam()
        E_interaction = np.sum(np.abs(self.S * zeta_foam * self.temperature))
        
        # 3. The Ratio (Alpha)
        # We add the Vacuum Floor to the denominator to prevent singularity
        # This represents the Zero Point Energy that cannot be removed.
        return E_interaction / (E_kinetic + self.vacuum_floor)

    def step_physics(self):
        # --- QUANTUM EVOLUTION ---
        zeta_foam = self.get_zeta_foam()
        
        # Energy Landscape
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        local_E = np.abs(grad_x)**2 + np.abs(grad_y)**2
        
        # Effective Mass/Frequency
        omega_eff = self.omega_0 * (1 - 0.05 * local_E)
        
        # Diffusion
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # Force Terms
        # 1. Driving force from Zeta Foam (scaled by T)
        force_zeta = self.S * 1j * (zeta_foam * self.temperature)
        
        # 2. Restoring Force (Mass Term) - Keeps S finite
        force_mass = -self.mass * self.S
        
        # Update Field
        # Persistence 0.98 (High memory, stable vacuum)
        S_new = (self.S * 0.98) * np.exp(1j * omega_eff * 0.01) + \
                (0.01 * laplacian) + \
                (0.01 * force_zeta) + \
                (0.01 * force_mass)
        
        # Renormalize (Unitary evolution)
        self.S = S_new / np.linalg.norm(S_new)
