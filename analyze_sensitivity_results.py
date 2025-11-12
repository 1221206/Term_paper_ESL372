
import os
import re

def analyze_sensitivity_results(base_folder):
    """
    Analyzes the sensitivity analysis results to find the best hyperparameters.

    Args:
        base_folder (str): The base folder where the sensitivity analysis results are stored.
    """
    best_mse = float('inf')
    best_hyperparameters = None

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == 'experiment_summary.txt':
                summary_path = os.path.join(root, file)
                with open(summary_path, 'r') as f:
                    content = f.read()
                    mse_match = re.search(r'MSE: (\d+\.\d+)', content)
                    alpha_match = re.search(r'alpha: (\d+\.\d+)', content)
                    beta_match = re.search(r'beta: (\d+\.\d+)', content)

                    if mse_match and alpha_match and beta_match:
                        mse = float(mse_match.group(1))
                        alpha = float(alpha_match.group(1))
                        beta = float(beta_match.group(1))

                        if mse < best_mse:
                            best_mse = mse
                            best_hyperparameters = {'alpha': alpha, 'beta': beta}

    if best_hyperparameters:
        print(f"Best MSE: {best_mse}")
        print(f"Best hyperparameters: alpha={best_hyperparameters['alpha']}, beta={best_hyperparameters['beta']}")
    else:
        print("No results found.")

if __name__ == '__main__':
    analyze_sensitivity_results('MIT_sensitivity_analysis_without_L2_PDE')
