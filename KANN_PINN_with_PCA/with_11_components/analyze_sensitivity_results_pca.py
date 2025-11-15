
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    """Parses a log file to extract the best validation MSE and corresponding test metrics."""
    best_valid_mse = float('inf')
    best_test_metrics = {}
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if '[Valid]' in line:
                match_mse = re.search(r'MSE: ([\d.]+)', line)
                if match_mse:
                    current_valid_mse = float(match_mse.group(1))
                    if current_valid_mse < best_valid_mse:
                        best_valid_mse = current_valid_mse
            elif '[Test]' in line and best_valid_mse != float('inf'):
                # This assumes the [Test] line immediately follows the [Valid] line for the best epoch
                # A more robust parsing would involve storing metrics per epoch and then selecting
                # based on the epoch with best_valid_mse. For simplicity, we'll take the last best test metrics.
                match_metrics = re.search(r'MSE: ([\d.]+), MAE: ([\d.]+), MAPE: ([\d.]+), RMSE: ([\d.]+), R2: ([-\d.]+)', line)
                if match_metrics:
                    best_test_metrics = {
                        'MSE': float(match_metrics.group(1)),
                        'MAE': float(match_metrics.group(2)),
                        'MAPE': float(match_metrics.group(3)),
                        'RMSE': float(match_metrics.group(4)),
                        'R2': float(match_metrics.group(5))
                    }
    return best_valid_mse, best_test_metrics

def parse_config_file(config_file_path):
    """Parses a config.txt file to extract hyperparameters."""
    params = {}
    with open(config_file_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(': ', 1)
                try:
                    params[key] = float(value) if '.' in value or 'e' in value else int(value)
                except ValueError:
                    params[key] = value
    return params

def main():
    base_sensitivity_dir = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/sensitivity_analysis_hybrid'
    results_dir = os.path.join(base_sensitivity_dir, 'analysis_plots')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # Iterate through each sensitivity folder (e.g., lambda1_sensitivity, lambda2_sensitivity)
    for param_sensitivity_folder in os.listdir(base_sensitivity_dir):
        param_sensitivity_path = os.path.join(base_sensitivity_dir, param_sensitivity_folder)
        if os.path.isdir(param_sensitivity_path):
            param_name = param_sensitivity_folder.replace('_sensitivity', '')

            # Iterate through each experiment folder within the sensitivity folder
            for exp_folder_name in os.listdir(param_sensitivity_path):
                exp_path = os.path.join(param_sensitivity_path, exp_folder_name)
                if os.path.isdir(exp_path):
                    log_file = os.path.join(exp_path, 'logging_pca_sensitivity.txt')
                    config_file = os.path.join(exp_path, 'config.txt')

                    if os.path.exists(log_file) and os.path.exists(config_file):
                        params = parse_config_file(config_file)
                        best_valid_mse, test_metrics = parse_log_file(log_file)
                        
                        if test_metrics: # Ensure metrics were found
                            result = {
                                'param_name': param_name,
                                'param_value': params.get(param_name, None),
                                'best_valid_mse': best_valid_mse,
                                **test_metrics
                            }
                            all_results.append(result)
    
    if not all_results:
        print("No sensitivity analysis results found. Please ensure the sensitivity analysis script ran successfully.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n--- Sensitivity Analysis Results Summary ---")
    print(results_df)

    # Identify optimal parameters for each analyzed lambda
    optimal_params = {}
    for param_name in ['lambda1', 'lambda2', 'lambda3']:
        param_df = results_df[results_df['param_name'] == param_name]
        if not param_df.empty:
            # Assuming lower MSE is better
            optimal_row = param_df.loc[param_df['MSE'].idxmin()]
            optimal_params[param_name] = {
                'value': optimal_row['param_value'],
                'MSE': optimal_row['MSE'],
                'R2': optimal_row['R2']
            }
            print(f"\nOptimal {param_name}: Value={optimal_params[param_name]['value']}, MSE={optimal_params[param_name]['MSE']:.6f}, R2={optimal_params[param_name]['R2']:.4f}")

    # Plotting results
    print("\n--- Generating Sensitivity Analysis Plots ---")
    for param_name in ['lambda1', 'lambda2', 'lambda3']:
        param_df = results_df[results_df['param_name'] == param_name].sort_values(by='param_value')
        if not param_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(param_df['param_value'], param_df['MSE'], marker='o', linestyle='-', color='blue', label='Test MSE')
            plt.plot(param_df['param_value'], param_df['R2'], marker='x', linestyle='--', color='red', label='Test R2')
            plt.title(f'Sensitivity Analysis for {param_name}')
            plt.xlabel(f'{param_name} Value')
            plt.ylabel('Metric Value')
            plt.xscale('log' if param_name == 'lambda3' else 'linear') # Log scale for lambda3
            plt.grid(True)
            plt.legend()
            plot_path = os.path.join(results_dir, f'{param_name}_sensitivity_plot.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved plot for {param_name} to {plot_path}")
            plt.close()

if __name__ == '__main__':
    main()
