import numpy as np
from lattice_dynamics import LatticeDynamics
import time

def main():
    print("=" * 70)
    print("   PROJECT ORPHEUS: RESONANCE LOCK SIMULATION")
    print("=" * 70)
    
    # The Fine Structure Constant
    TARGET_ALPHA = 1 / 137.035999 
    
    sim = LatticeDynamics(N=64)
    
    print(f"Target Alpha: {TARGET_ALPHA:.6f}")
    print(f"Mechanism:    Coherent Zeta-Wave Interaction")
    print("-" * 70)
    print(f"{'Epoch':<6} | {'Alpha':<10} | {'Error':<10} | {'Temp (T)':<10} | {'Status'}")
    print("-" * 70)
    
    history = []
    
    # Pre-warm
    for _ in range(50): sim.step_physics()

    for epoch in range(1, 1501):
        
        sim.step_physics()
        alpha, temp = sim.apply_control(TARGET_ALPHA)
        history.append(alpha)
        
        if epoch % 50 == 0:
            error = alpha - TARGET_ALPHA
            
            if abs(error) < 1e-5: status = "[LOCKED]"
            elif abs(error) < 1e-4: status = "Converged"
            elif error > 0: status = "Cooling"
            else: status = "Heating"
            
            print(f"{epoch:<6} | {alpha:.6f}   | {error:+.6f}   | {temp:.6f}     | {status}")
            
            # Check for stability
            if epoch > 300:
                recent = history[-20:]
                if np.std(recent) < 1e-6 and abs(np.mean(recent) - TARGET_ALPHA) < 1e-5:
                    print("\n>>> RESONANCE ACHIEVED.")
                    print(f">>> The vacuum state has locked to Alpha = {np.mean(recent):.9f}")
                    break
                    
    final_alpha = history[-1]
    print("=" * 70)
    print(f"Final Alpha: {final_alpha:.9f}")
    print(f"Target:      {TARGET_ALPHA:.9f}")
    print(f"Deviation:   {abs(final_alpha - TARGET_ALPHA)/TARGET_ALPHA*100:.4f}%")

if __name__ == "__main__":
    main()
