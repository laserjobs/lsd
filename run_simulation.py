import numpy as np
from lattice_dynamics import LatticeDynamics
import time

def main():
    print("=" * 70)
    print("   PROJECT ORPHEUS: VACUUM RESONANCE (ROBUST)")
    print("=" * 70)
    
    TARGET_ALPHA = 1 / 137.035999
    MAX_EPOCHS = 1000
    
    print(f"Target Alpha:    {TARGET_ALPHA:.6f}")
    print(f"Initial T:       0.0075 (Soft Start)")
    print(f"Vacuum Floor:    0.0001")
    print("-" * 70)
    
    sim = LatticeDynamics(N=64)
    
    print(f"{'Epoch':<6} | {'Alpha':<10} | {'Error':<10} | {'Temp (T)':<10} | {'Status'}")
    print("-" * 70)
    
    history = []
    
    # Warmup
    for _ in range(20): sim.step_physics()

    for epoch in range(1, MAX_EPOCHS + 1):
        sim.step_physics()
        alpha, temp = sim.apply_control(TARGET_ALPHA)
        history.append(alpha)
        
        if epoch % 25 == 0:
            error = alpha - TARGET_ALPHA
            
            # Status logic
            if abs(error) < 1e-6: status = "[LOCKED]"
            elif abs(error) < 1e-4: status = "Converged"
            elif error > 0: status = "Damping"
            else: status = "Pumping"
            
            print(f"{epoch:<6} | {alpha:.6f}   | {error:+.6f}   | {temp:.6f}     | {status}")
            
            # Convergence criteria
            if epoch > 200:
                recent_avg = np.mean(history[-20:])
                recent_std = np.std(history[-20:])
                if abs(recent_avg - TARGET_ALPHA) < 5e-6 and recent_std < 5e-6:
                    print("\n>>> RESONANCE ACHIEVED.")
                    print(f">>> System locked at Alpha = {recent_avg:.9f}")
                    print(f">>> Equilibrium Temperature = {temp:.9f}")
                    break

    final_alpha = history[-1]
    print("=" * 70)
    print(f"Final Alpha: {final_alpha:.9f}")
    print(f"Target:      {TARGET_ALPHA:.9f}")
    print(f"Deviation:   {abs(final_alpha - TARGET_ALPHA)/TARGET_ALPHA*100:.4f}%")

if __name__ == "__main__":
    main()
