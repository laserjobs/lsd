import numpy as np

class LatticeDynamics:
    def __init__(self, N=16):
        self.N = N
        # Initialize grid with high-energy random noise (Primordial State)
        self.grid = np.random.normal(0, 1, (N, N, N, N))
        
        # State tracking for the controller
        self.prev_alpha = self.measure_coupling_ratio()
        self.integral_error = 0.0
        
    def measure_coupling_ratio(self):
        """
        Measures the 'Alpha' of the current grid state.
        Alpha ~ Variance / Mean_Energy (A simplified dimensionless observable)
        """
        variance = np.var(self.grid)
        mean_sq = np.mean(self.grid**2)
        if mean_sq == 0: return 0
        
        # This ratio represents the 'clumpiness' vs 'smoothness' of the field
        return variance / (mean_sq + 1e-9)

    def step_critical_damping(self, target_alpha):
        """
        Applies a force to the lattice to drive Alpha -> Target
        using Critical Damping (Zeta = 0.707).
        """
        current_alpha = self.measure_coupling_ratio()
        
        # 1. Calculate Error (Displacement from Vacuum)
        error = current_alpha - target_alpha
        
        # 2. Calculate Velocity (Rate of change of Alpha)
        velocity = current_alpha - self.prev_alpha
        
        # 3. Define Control Physics
        # Stiffness (k): How hard we pull towards the vacuum. 
        # We need a stiff field to overcome the natural 0.26 geometry.
        k_stiffness = 0.1 
        
        # Damping (c): Derived from Zeta = 0.707 (Critical Damping)
        # Formula: c = Zeta * 2 * sqrt(k)
        zeta = 0.707
        c_damping = zeta * 2 * np.sqrt(k_stiffness)
        
        # 4. Calculate Feedback Force
        # Force = -k*x - c*v
        force = - (k_stiffness * error) - (c_damping * velocity)
        
        # 5. Apply Force to the Grid
        # We scale the grid energy up or down based on the force.
        # If Alpha is too high, we need to 'smooth' the grid (reduce variance).
        # If Alpha is too low, we roughen it.
        
        scaling_factor = 1.0 + force
        
        # Clamp scaling to prevent numerical explosion
        scaling_factor = np.clip(scaling_factor, 0.8, 1.2)
        
        # Apply the "Cooling" or "Heating"
        # To lower alpha (reduce variance relative to mean), we mix the grid towards its mean.
        if error > 0: # Alpha too high, need to cool
            self.grid = self.grid * scaling_factor
        else:         # Alpha too low, need to excite
            noise = np.random.normal(0, 0.01, (self.N, self.N, self.N, self.N))
            self.grid += noise
            
        # Re-normalize slightly to prevent total energy collapse
        # (Simulating energy conservation in the background)
        current_energy = np.sum(self.grid**2)
        if current_energy > 0:
            self.grid /= np.sqrt(current_energy / (self.N**4))

        # Store state
        self.prev_alpha = current_alpha
