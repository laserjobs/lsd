import numpy as np

class LatticeDynamics:
    def __init__(self, N=16):
        self.N = N
        # Initialize with a "Hot" broken state (High Energy Plasma)
        # High variance (10.0) ensures we start at Alpha ~ 1.0
        self.grid = np.random.normal(loc=1.0, scale=10.0, size=(N, N, N, N))
        
        # PID Controller State
        self.prev_error = 0.0
        self.integral_error = 0.0
        
    def measure_coupling_ratio(self):
        """
        Alpha = Variance / Total_Energy
        """
        variance = np.var(self.grid)
        mean_sq = np.mean(self.grid**2)
        
        if mean_sq == 0: return 1.0
        return variance / mean_sq

    def step_symmetry_breaking(self, target_alpha, zeta=0.707):
        """
        Applies Critical Damping (PID Control) to the lattice fluctuations.
        Forces the system to condense from Alpha ~ 1.0 down to Target.
        """
        current_alpha = self.measure_coupling_ratio()
        
        # 1. Calculate Error
        error = current_alpha - target_alpha
        
        # 2. PID Terms
        # Proportional
        Kp = 0.5
        
        # Integral (Accumulates error to close the steady-state gap)
        self.integral_error += error
        Ki = 0.02 
        
        # Derivative (Damping)
        derivative = error - self.prev_error
        Kd = zeta * 2 * np.sqrt(Kp) # Critical Damping Relation
        
        # 3. Calculate Damping Force
        # F = Kp*e + Ki*int + Kd*de/dt
        damping_force = (Kp * error) + (Ki * self.integral_error) + (Kd * derivative)
        
        # 4. Apply Physics (Cooling/Heating)
        # We separate the Vacuum (Mean) from the Fluctuations (Noise)
        vacuum_mean = np.mean(self.grid)
        fluctuations = self.grid - vacuum_mean
        
        # Calculate Scaling Factor (1.0 = No Change, <1.0 = Cool, >1.0 = Heat)
        # We clamp the force to prevent numerical explosions
        scaling_factor = 1.0 - np.clip(damping_force, -0.5, 0.5)
        
        # Apply scaling ONLY to the fluctuations
        self.grid = vacuum_mean + (fluctuations * scaling_factor)
        
        # 5. Vacuum Maintenance
        # Ensure the Vacuum Expectation Value (VEV) doesn't collapse to zero
        if abs(vacuum_mean) < 1.0:
             self.grid += 0.01 * np.sign(vacuum_mean) if abs(vacuum_mean) > 1e-9 else 0.01

        # Update State
        self.prev_error = error
        
        return current_alpha, damping_force
