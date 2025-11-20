import numpy as np
import time
from lattice_dynamics import LatticeDynamics

def main():
    print("-" * 60)
    print("   LATTICE SPECTRAL DYNAMICS: VACUUM RELAXATION SIMULATION")
    print("-" * 60)

    # 1. Initialize the Physics Engine
    try:
        sim = LatticeDynamics()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 2. Simulation Parameters
    MAX_EPOCHS = 1000
    TARGET_ALPHA = 1 / 137.035999
    TOLERANCE = 0.05 # 5% tolerance for this low-res prototype
    
    print(f"Target Coupling Constant: {TARGET_ALPHA:.6f}")
    print("Starting evolution loop...")

    history = []
    start_time = time.time()

    # 3. Main Loop
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            # Advance physics
            sim.step()
            
            # Measure and report every 10 steps
            if epoch % 10 == 0:
                alpha = sim.measure_coupling_ratio()
                history.append(alpha)
                
                # Calculate running average to smooth noise
                avg_alpha = np.mean(history[-5:]) if len(history) >= 5 else alpha
                
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch:3d}] Emergent Alpha: {avg_alpha:.6f} (t={elapsed:.1f}s)")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    # 4. Final Analysis
    final_alpha = np.mean(history[-10:]) # Stable average of last 10 measurements
    deviation = abs(final_alpha - TARGET_ALPHA) / TARGET_ALPHA * 100

    print("-" * 60)
    print("SIMULATION COMPLETE")
    print("-" * 60)
    print(f"Converged Value: {final_alpha:.6f}")
    print(f"Target Value:    {TARGET_ALPHA:.6f}")
    print(f"Deviation:       {deviation:.2f}%")
    
    if deviation < 5.0:
        print("\n>>> RESULT: CONVERGENCE DETECTED.")
        print(">>> The system has self-organized to the physical coupling constant.")
    else:
        print("\n>>> RESULT: DIVERGENCE.")
        print(">>> Grid size (N) may be too small or cooling rate requires tuning.")
        
if __name__ == "__main__":
    main()
