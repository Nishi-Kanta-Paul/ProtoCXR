"""Compare baseline results and generate TABLE I + Fig 3."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.evaluate import evaluate_all_models, save_table1
from src.figures import fig_auc_comparison


def main() -> None:
    """Run baseline comparison workflow.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    config = Config()
    experiments_dir = os.path.join(config.DRIVE_ROOT, "experiments")
    results = evaluate_all_models(experiments_dir, config)

    save_table1(results, config.TABLES_DIR)
    fig_auc_comparison(results, config.FIGURES_DIR, config)

    table_path = os.path.join(config.TABLES_DIR, "table1_auc.txt")
    if os.path.exists(table_path):
        with open(table_path, "r", encoding="utf-8") as file_obj:
            print(file_obj.read())
    print(f"Saved table: {table_path}")
    print(f"Saved figure: {os.path.join(config.FIGURES_DIR, 'fig3_auc_comparison.png')}")


if __name__ == "__main__":
    main()