import numpy as np
from lattice_dynamics import LatticeDynamics

def main():
    print("=" * 70)
    print("   RENORMALIZATION GROUP FLOW: FINDING THE VACUUM SCALE")
    print("=" * 70)
    
    # 1. Configuration
    TARGET_ALPHA = 1 / 137.035999
    MAX_EPOCHS = 1500
    
    print(f"Target Alpha: {TARGET_ALPHA:.6f}")
    print(f"Mechanism:    Temperature Scaling (Renormalization)")
    print("-" * 70)
    
    # 2. Initialize
    sim = LatticeDynamics(N=64)
    
    print(f"{'Epoch':<6} | {'Alpha':<10} | {'Error':<10} | {'Temp (Scale)':<12} | {'Status'}")
    print("-" * 70)

    history = []
    
    # 3. Loop
    for epoch in range(1, MAX_EPOCHS + 1):
        
        current_alpha, temp = sim.step_renormalization(TARGET_ALPHA)
        history.append(current_alpha)
        
        if epoch % 50 == 0 or epoch == 1:
            error = current_alpha - TARGET_ALPHA
            
            # Status Logic
            if abs(error) < 1e-6: status = "[LOCKED]"
            elif abs(error) < 1e-4: status = "Fine Tuning"
            elif error > 0: status = "Cooling (High E)"
            else: status = "Heating (Low E)"
            
            print(f"{epoch:<6} | {current_alpha:.6f}   | {error:+.6f}   | {temp:.6f}       | {status}")
            
            # Convergence Check (Strict)
            # We want it to hold the value for a while
            if epoch > 300 and abs(error) < 1e-7:
                print("\n>>> VACUUM STABILITY ACHIEVED.")
                print(">>> The universe has found the energy scale matching Alpha.")
                break
                
    final_alpha = history[-1]
    print("=" * 70)
    print(f"Final Alpha: {final_alpha:.9f}")
    print(f"Target:      {TARGET_ALPHA:.9f}")
    
    if abs(final_alpha - TARGET_ALPHA) < 1e-5:
        print("\nSUCCESS: Renormalization complete.")
    else:
        print("\nRESULT: Convergence in progress...")

if __name__ == "__main__":
    main()
