"""
baselines/compare_results.py
============================
Loads results.json from all experiment directories and produces
Table 1 and Fig 3 comparisons.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.evaluate import evaluate_all_models, save_table1
from src.figures import fig_auc_comparison


def main():
    config = Config()
    results_dir = os.path.join(config.DRIVE_ROOT, "experiments")
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Please run experiments first.")
        return
        
    print(f"Loading results from {results_dir}...")
    results_dict = evaluate_all_models(results_dir, config)
    
    print("\nResults comparison gathered:")
    for model_name, res in results_dict.items():
        print(f"  {model_name}: {res}")
        
    save_table1(results_dict, config.TABLES_DIR)
    fig_auc_comparison(results_dict, config.FIGURES_DIR, config)
    
    print("\nComparison complete. Check outputs directory.")


if __name__ == "__main__":
    main()
