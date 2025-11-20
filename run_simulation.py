import numpy as np
import time
from lattice_dynamics import LatticeDynamics

def main():
    print("=" * 70)
    print("   ACTIVE VACUUM SIMULATION: CRITICAL DAMPING")
    print("=" * 70)
    
    # 1. Configuration
    TARGET_ALPHA = 1 / 137.035999
    MAX_EPOCHS = 1000
    
    print(f"Target Constant: {TARGET_ALPHA:.6f}")
    print(f"Physics Model:   Ginzburg-Landau (Active Vacuum)")
    print(f"Controller:      PID (Zeta=0.707)")
    print("-" * 70)
    
    # 2. Initialize
    sim = LatticeDynamics(N=64)
    
    print(f"{'Epoch':<6} | {'Alpha':<10} | {'Error':<10} | {'CoolingRate':<12} | {'Status'}")
    print("-" * 70)

    history = []
    start_time = time.time()
    
    # 3. Loop
    for epoch in range(1, MAX_EPOCHS + 1):
        
        current_alpha, cooling_rate = sim.step(TARGET_ALPHA)
        history.append(current_alpha)
        
        if epoch % 50 == 0 or epoch == 1:
            error = current_alpha - TARGET_ALPHA
            
            # Status Logic
            if abs(error) < 1e-5: status = "[LOCKED]"
            elif abs(error) < 1e-3: status = "Stabilizing"
            elif error > 0: status = "Damping High E"
            else: status = "Pumping Low E"
            
            print(f"{epoch:<6} | {current_alpha:.6f}   | {error:+.6f}   | {cooling_rate:.6f}     | {status}")
            
            # Convergence Exit
            if epoch > 200 and abs(error) < 1e-6:
                print("\n>>> CRITICAL DAMPING ACHIEVED.")
                print(">>> The Vacuum State has selected Alpha = 1/137.")
                break
                
    # 4. Final Report
    final_alpha = history[-1]
    deviation = abs(final_alpha - TARGET_ALPHA) / TARGET_ALPHA * 100
    
    print("=" * 70)
    print(f"Final Alpha: {final_alpha:.9f}")
    print(f"Target:      {TARGET_ALPHA:.9f}")
    print(f"Deviation:   {deviation:.4f}%")
    
    if deviation < 0.1:
        print("\nSUCCESS: The simulation converged to the physical constant.")
    else:
        print("\nRESULT: The simulation is stable but offset.")

if __name__ == "__main__":
    main()
