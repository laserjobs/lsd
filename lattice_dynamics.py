import numpy as np

class LatticeDynamics:
    def __init__(self, N=64):
        self.N = N
        # 1. Initialize Complex Field (The Vacuum)
        self.S = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        self.S /= np.linalg.norm(self.S)
        
        # 2. generate the Base Noise Pattern (Riemann Zeros)
        # We calculate the SHAPE here, but the AMPLITUDE (Temperature) is dynamic.
        t = np.linspace(0, 100, N)
        zeros = [14.13, 21.02, 25.01] 
        noise_1d = np.sum([np.sin(z * t) for z in zeros], axis=0)
        self.Theta_Base = np.outer(noise_1d, noise_1d)
        # Normalize base pattern to unit range [-1, 1]
        self.Theta_Base /= np.max(np.abs(self.Theta_Base))

        # 3. Physics State
        self.temperature = 0.05  # Initial "Energy Scale" of the universe
        self.omega_0 = 1.0
        self.kappa = 0.05
        
        # 4. Controller State
        self.prev_error = 0.0
        self.integral_error = 0.0

    def measure_alpha(self):
        """
        Alpha = Interaction_Energy / Kinetic_Energy
        """
        # Kinetic Energy (Gradient/Stiffness)
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        E_kinetic = np.sum(np.abs(grad_x)**2 + np.abs(grad_y)**2)
        
        # Interaction Energy (Coupling to the Noise Field)
        # Note: The interaction strength depends on Temperature
        interaction_field = self.Theta_Base * self.temperature
        E_interaction = np.sum(np.abs(self.S * interaction_field))
        
        if E_kinetic < 1e-9: return 1.0 # Prevent divide by zero if frozen
        
        return E_interaction / E_kinetic

    def step_renormalization(self, target_alpha, zeta=0.707):
        # --- A. QUANTUM EVOLUTION ---
        # 1. Calculate Energy Landscape
        grad_x = np.roll(self.S, 1, axis=0) - self.S
        grad_y = np.roll(self.S, 1, axis=1) - self.S
        local_E = np.abs(grad_x)**2 + np.abs(grad_y)**2
        
        # 2. Effective Frequency
        omega_eff = self.omega_0 * (1 - self.kappa * local_E)
        
        # 3. Diffusion (Laplacian)
        laplacian = (np.roll(self.S, 1, axis=0) + np.roll(self.S, -1, axis=0) + 
                     np.roll(self.S, 1, axis=1) + np.roll(self.S, -1, axis=1) - 4*self.S)
        
        # 4. Interaction Term (Scaled by Temperature)
        noise_term = self.S * 1j * (self.Theta_Base * self.temperature)
        
        # 5. Update Step (Ginzburg-Landau)
        # Fixed cooling rate for physics (0.95)
        S_new = (self.S * 0.95) * np.exp(1j * omega_eff * 0.01) + \
                (0.01 * laplacian) + \
                (0.01 * noise_term)
        
        # Renormalize Field
        self.S = S_new / np.linalg.norm(S_new)

        # --- B. RENORMALIZATION CONTROL (Running Coupling) ---
        # We adjust the Temperature to find the scale where Alpha matches Target
        current_alpha = self.measure_alpha()
        error = current_alpha - target_alpha
        
        # PID Tuning for Temperature
        # If Alpha is too high -> Reduce Temperature (Cool Down)
        # If Alpha is too low  -> Increase Temperature (Heat Up)
        
        Kp = 0.1
        Ki = 0.005
        Kd = zeta * 2 * np.sqrt(Kp) # Critical Damping
        
        self.integral_error += error
        d_error = error - self.prev_error
        
        # Adjustment to Temperature
        # Note: Positive error (Alpha too high) requires Negative adjustment (Cooling)
        adjustment = (Kp * error) + (Ki * self.integral_error) + (Kd * d_error)
        
        self.temperature -= adjustment * 0.01 # Scaling factor for stability
        
        # Clamp Temperature (Cannot be negative, cannot be infinite)
        # Lower bound ensures we don't hit absolute zero and divide by zero
        self.temperature = np.clip(self.temperature, 1e-6, 1.0)
        
        self.prev_error = error
        
        return current_alpha, self.temperature
