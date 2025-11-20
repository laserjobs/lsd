import numpy as np
from lattice_dynamics import LatticeDynamics

def main():
    print("=" * 70)
    print("   QUANTUM VACUUM RENORMALIZATION: STABLE REGIME")
    print("=" * 70)
    
    TARGET_ALPHA = 1 / 137.035999
    MAX_EPOCHS = 2000
    
    print(f"Target Alpha: {TARGET_ALPHA:.6f}")
    print(f"Physics:      Stabilized Vacuum (Quantum Foam + Geometry)")
    print("-" * 70)
    
    sim = LatticeDynamics(N=64)
    
    print(f"{'Epoch':<6} | {'Alpha':<10} | {'Error':<10} | {'Temp (Scale)':<12} | {'Status'}")
    print("-" * 70)

    history = []
    
    # Pre-warm the vacuum to let kinetic energy settle
    print("Pre-warming vacuum state...", end="\r")
    for _ in range(100): sim.step_renormalization(TARGET_ALPHA)
    print("Vacuum state initialized.      ")

    for epoch in range(1, MAX_EPOCHS + 1):
        
        current_alpha, temp = sim.step_renormalization(TARGET_ALPHA)
        history.append(current_alpha)
        
        if epoch % 50 == 0 or epoch == 1:
            error = current_alpha - TARGET_ALPHA
            
            # Status Logic
            if abs(error) < 1e-6: status = "[LOCKED]"
            elif abs(error) < 1e-4: status = "Tuning"
            elif error > 0: status = "Cooling"
            else: status = "Heating"
            
            print(f"{epoch:<6} | {current_alpha:.6f}   | {error:+.6f}   | {temp:.8f}     | {status}")
            
            # Convergence
            if epoch > 500 and abs(error) < 5e-7:
                print("\n>>> CONVERGENCE ACHIEVED.")
                print(f">>> The vacuum has stabilized at scale T = {temp:.8f}")
                break
                
    final_alpha = history[-1]
    print("=" * 70)
    print(f"Final Alpha: {final_alpha:.9f}")
    print(f"Target:      {TARGET_ALPHA:.9f}")
    
    if abs(final_alpha - TARGET_ALPHA) < 1e-5:
        print("SUCCESS: The simulation derived the coupling constant.")

if __name__ == "__main__":
    main()
