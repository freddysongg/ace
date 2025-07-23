#!/usr/bin/env python3
"""
Quick script to run MLP evaluation only mode after fixing the JSON serialization issue.
This will load the trained models and run evaluation without retraining.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the MLP evaluation in per-pollutant mode."""
    
    print("🔍 MLP Per-Pollutant Evaluation Script")
    print("=" * 50)
    
    results_dir = Path("results/mlp-per-pollutant")
    model_paths = [
        results_dir / "Ozone" / "mlp_model.keras",
        results_dir / "PM2.5" / "mlp_model.keras", 
        results_dir / "NO2" / "mlp_model.keras"
    ]
    
    models_exist = all(path.exists() for path in model_paths)
    
    if models_exist:
        print("✅ Found existing trained models:")
        for path in model_paths:
            print(f"   - {path}")
        print("\n🚀 Running evaluation only (no training)...")
        cmd = [
            sys.executable, "single_model.py",
            "--model", "mlp",
            "--eval-only"
        ]
    else:
        print("❌ Models not found. Missing models:")
        for path in model_paths:
            if not path.exists():
                print(f"   - {path}")
        print("\n🏋️  Running full training and evaluation...")
        cmd = [
            sys.executable, "single_model.py", 
            "--model", "mlp"
        ]
    
    print(f"\n💻 Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 50)
        print("✅ Evaluation completed successfully!")
        print("\n📊 Results available in:")
        print("   - results/mlp-per-pollutant/comparison_metrics_summary.json")
        print("   - results/mlp-per-pollutant/{Ozone,PM2.5,NO2}/")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)