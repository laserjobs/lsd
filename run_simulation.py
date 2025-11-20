import numpy as np
import time
# This import works because the file above is named lattice_dynamics.py
from lattice_dynamics import LatticeDynamics

def main():
    print("=" * 70)
    print("   SYMMETRY BREAKING: CRITICAL DAMPING SIMULATION")
    print("=" * 70)
    
    # 1. Configuration
    TARGET_ALPHA = 1 / 137.036  # ~0.007297
    ZETA_TARGET = 0.707         # Critical Damping
    MAX_EPOCHS = 600            # Enough time for Integral term to settle
    
    print(f"Target Coupling (Alpha): {TARGET_ALPHA:.6f}")
    print(f"Damping Ratio (Zeta):    {ZETA_TARGET:.3f}")
    print("Initializing Primordial High-Energy State...")
    
    sim = LatticeDynamics(N=16)
    
    # Verify starting conditions
    start_alpha = sim.measure_coupling_ratio()
    print(f"Primordial Alpha:        {start_alpha:.4f} (Chaotic/Hot)")
    print("-" * 70)
    print(f"{'Epoch':<6} | {'Alpha':<10} | {'Error':<10} | {'DampForce':<10} | {'State'}")
    print("-" * 70)

    history = []
    
    # 2. Evolution Loop
    for epoch in range(1, MAX_EPOCHS + 1):
        
        # Execute Physics Step
        current_alpha, force = sim.step_symmetry_breaking(TARGET_ALPHA, zeta=ZETA_TARGET)
        history.append(current_alpha)
        
        if epoch % 20 == 0 or epoch == 1:
            error = current_alpha - TARGET_ALPHA
            
            # Determine State Label
            if current_alpha > 0.1:
                state = "PLASMA"
            elif current_alpha > 0.01:
                state = "COOLING"
            elif abs(error) < 0.0001:
                state = "VACUUM"
            else:
                state = "STABLE"
                
            print(f"{epoch:<6} | {current_alpha:.6f}   | {error:+.6f}   | {force:+.4f}     | {state}")
            
            # Convergence Check (Strict)
            if epoch > 50 and abs(error) < 1e-6:
                print("\n>>> CONVERGENCE REACHED.")
                print(f">>> System locked to {current_alpha:.9f}")
                break

    # 3. Analysis
    final_alpha = history[-1]
    print("-" * 70)
    print(f"Final Alpha: {final_alpha:.9f}")
    print(f"Target:      {TARGET_ALPHA:.9f}")
    
    if abs(final_alpha - TARGET_ALPHA) < 1e-5:
        print("\nRESULT: SUCCESS.")
        print("The Universe has cooled and stabilized at the Fine Structure Constant.")
    else:
        print("\nRESULT: INCOMPLETE.")
        print("The system is stable but has not yet fully converged.")

if __name__ == "__main__":
    main()
