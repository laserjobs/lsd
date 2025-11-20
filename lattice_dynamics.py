import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        # 1. Initialize Complex Quantum Field (2D for clear wave dynamics)
        # Random amplitude + Random phase
        self.S = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        self.S /= np.linalg.norm(self.S)
        
        # 2. Initialize Noise Kernel (The "Pump" or "Vacuum Energy")
        # This prevents the universe from freezing to absolute zero.
        t = np.linspace(0, 100, N)
        # Zeros of Zeta function (Simulating prime-number based resonance)
        zeros = [14.13, 21.02, 25.01] 
        noise_1d = np.sum([np.sin(z * t) for z in zeros], axis=0)
        self.Theta = np.outer(noise_1d, noise_1d)
        # Normalize driving force
        self.Theta /= np.max(np.abs(self.Theta)) * 5.0 

        # 3. Physics Constants
        self.omega_0 = 1.0
        self.coupling = 0.01 
        self.kappa = 0.05
        
        # 4. Control State
        self.cooling_rate = 0.90 # The variable we tune to find Alpha
        self.prev_error = 0.0
        self.integral_error = 0.0

    def measure_alpha(self):
        """
        Alpha ~ Interaction_Energy / Kinetic_Energy
        """
        # Gradient energy (Kinetic)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2)
        
        # Interaction with Vacuum (Potential)
        E_interaction = np.sum(np.abs(self.S * self.Theta))
        
        if E_kinetic == 0: return 0.0
        
        # This ratio defines the "Structure" of the universe
        return E_interaction / (E_kinetic + 1e-9)

    def step(self, target_alpha, zeta=0.707):
        # --- A. QUANTUM EVOLUTION (The Physics) ---
        # 1. Calculate Local Energy
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        local_E = np.abs(grad_x)**2 + np.abs(grad_y)**2
        
        # 2. Effective Frequency (Non-linear term)
        omega_eff = self.omega_0 * (1 - self.kappa * local_E)
        
        # 3. Laplacian (Diffusion)
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # 4. Update Field
        # New State = (Decay) * (Rotation) + (Diffusion) + (Vacuum Drive)
        S_new = (self.S * self.cooling_rate) * np.exp(1j * omega_eff * 0.01) + \
                (self.coupling * laplacian * 0.01) + \
                (self.S * 1j * self.Theta * 0.01)
        
        # Renormalize (Constraint of Unitary Evolution)
        self.S = S_new / np.linalg.norm(S_new)

        # --- B. FEEDBACK CONTROL (The Tuning) ---
        current_alpha = self.measure_alpha()
        error = current_alpha - target_alpha
        
        # PID Controller for Cooling Rate
        # We adjust the global cooling rate to force Alpha to the target.
        Kp = 0.5
        Kd = 1.0 # Critical Damping (Derived for Zeta=0.707)
        Ki = 0.05 # Integral term to fix steady-state error
        
        self.integral_error += error
        d_error = error - self.prev_error
        
        # Adjustment = Force applied to the variable "cooling_rate"
        adjustment = (Kp * error) + (Kd * d_error) + (Ki * self.integral_error)
        
        # Apply adjustment (Slowly)
        self.cooling_rate -= adjustment * 0.01
        
        # Clamp to physical bounds (0.5 = Rapid Cooling, 1.0 = No Cooling)
        self.cooling_rate = np.clip(self.cooling_rate, 0.5, 0.999)
        
        self.prev_error = error
        
        return current_alpha, self.cooling_rate
