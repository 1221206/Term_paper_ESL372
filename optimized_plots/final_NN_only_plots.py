import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import MyMITdata
from Model.Model_NN import Trainer
from utils.util import eval_metrix

def get_args_for_trainer():
    parser = argparse.ArgumentParser('Hyper Parameters for NN model')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='optimized_plots/NN_Final_Optimized_Results', help='save folder')
    return parser.parse_args([]) # Parse empty list to get default args

def load_MyMIT_data(args):
    root = r'C:\Users\pc\Inf Hyp\Power system\Pi_KANN\MIT_data'
    train_list, test_list = [], []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    data = MyMITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': trainloader['train_2'], 'valid': trainloader['valid_2'], 'test': testloader['test_3']}
    return dataloader

def plot_single_experiment_losses(experiment_folder, param_name, param_value):
    """
    Loads epoch losses from an experiment folder and plots them.
    """
    losses_path = os.path.join(experiment_folder, 'epoch_losses.npy')
    
    if not os.path.exists(losses_path):
        print(f"Error: epoch_losses.npy not found in {experiment_folder}")
        return

    losses = np.load(losses_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Epoch Loss', color='blue')
    plt.title(f'NN Sensitivity: {param_name} = {param_value} (Epoch Loss)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Epoch Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plot_path = os.path.join(experiment_folder, f'{param_name}_{param_value}_epoch_loss.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Epoch loss plot saved to: {plot_path}")
    plt.close() # Close the plot to free up memory

def analyze_and_plot_sensitivity_type(base_output_folder, sensitivity_type_folder, param_name):
    """
    Analyzes results for a specific sensitivity type, generates aggregated plots and summary files.
    Returns a list of dictionaries with param_name, param_value, and min_loss.
    """
    sensitivity_type_path = os.path.join(base_output_folder, sensitivity_type_folder)
    
    param_results = []
    
    print(f"Analyzing {param_name} sensitivity...")

    for experiment_folder_name in sorted(os.listdir(sensitivity_type_path)):
        experiment_path = os.path.join(sensitivity_type_path, experiment_folder_name)
        if os.path.isdir(experiment_path):
            param_value_str = experiment_folder_name.replace(f'{param_name}_', '')
            try:
                param_value = float(param_value_str)
            except ValueError:
                print(f"Warning: Could not convert '{param_value_str}' to float. Skipping.")
                continue
            
            losses_path = os.path.join(experiment_path, 'epoch_losses.npy')
            
            if os.path.exists(losses_path):
                losses = np.load(losses_path)
                if len(losses) > 0:
                    min_loss = np.min(losses)
                    param_results.append({
                        'param_name': param_name,
                        'param_value': param_value,
                        'min_loss': min_loss
                    })
                    print(f"  {param_name} = {param_value}: Min Loss = {min_loss:.6f}")
                    plot_single_experiment_losses(experiment_path, param_name, param_value)
                else:
                    print(f"Warning: epoch_losses.npy is empty in {experiment_path}. Skipping.")
            else:
                print(f"Error: epoch_losses.npy not found in {experiment_path}. Skipping.")

    if not param_results:
        print(f"No valid data found for {param_name} sensitivity analysis.")
        return []

    # Sort by parameter value for consistent plotting
    param_values = [res['param_value'] for res in param_results]
    min_losses = [res['min_loss'] for res in param_results]
    sorted_indices = np.argsort(param_values)
    param_values = np.array(param_values)[sorted_indices]
    min_losses = np.array(min_losses)[sorted_indices]

    # --- Generate Aggregated Plot ---
    analysis_results_folder = os.path.join(base_output_folder, 'analysis_results')
    os.makedirs(analysis_results_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, min_losses, marker='o', linestyle='-', color='green')
    plt.title(f'NN Sensitivity Analysis: Min Epoch Loss vs. {param_name}', fontsize=16)
    plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Minimum Epoch Loss', fontsize=12)
    plt.xscale('log') if param_name == 'l2_lambda' else None
    plt.grid(True)
    
    aggregated_plot_path = os.path.join(analysis_results_folder, f'{param_name}_sensitivity_analysis.png')
    plt.savefig(aggregated_plot_path, dpi=300)
    print(f"Aggregated sensitivity plot saved to: {aggregated_plot_path}")
    plt.close()

    # --- Generate Summary Text File ---
    summary_txt_path = os.path.join(analysis_results_folder, f'{param_name}_sensitivity_analysis.txt')
    with open(summary_txt_path, 'w') as f:
        f.write(f"NN Sensitivity Analysis for: {param_name}\n")
        f.write("--------------------------------------------------\n")
        for i in range(len(param_values)):
            f.write(f"{param_name}: {param_values[i]}, Minimum Epoch Loss: {min_losses[i]:.6f}\n")
    print(f"Summary text file saved to: {summary_txt_path}")

    # --- Generate CSV File ---
    summary_df = pd.DataFrame({param_name: param_values, 'Min_Epoch_Loss': min_losses})
    csv_path = os.path.join(analysis_results_folder, f'{param_name}_sensitivity.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary CSV file saved to: {csv_path}")

    return param_results

def train_final_optimal_nn_model(optimal_hyperparams, final_save_folder):
    print("\n--- Starting Final NN Model Training with Optimal Hyperparameters ---")
    if not os.path.exists(final_save_folder):
        os.makedirs(final_save_folder)

    args = get_args_for_trainer()
    setattr(args, 'l2_lambda', optimal_hyperparams.get('l2_lambda', args.l2_lambda))
    setattr(args, 'u_hidden_dim', int(optimal_hyperparams.get('u_hidden_dim', args.u_hidden_dim)))
    setattr(args, 'u_layers_num', int(optimal_hyperparams.get('u_layers_num', args.u_layers_num)))
    setattr(args, 'save_folder', final_save_folder)
    setattr(args, 'log_dir', 'logging.txt')

    dataloader = load_MyMIT_data(args)
    trainer = Trainer(args)
    trainer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
    print("--- Final NN Model Training Finished ---")

    # --- Load Results and Generate Plots ---
    print("--- Generating Final Plots and Metrics for Optimal NN Model ---")
    pred_label_path = os.path.join(final_save_folder, 'pred_label.npy')
    true_label_path = os.path.join(final_save_folder, 'true_label.npy')

    if not (os.path.exists(pred_label_path) and os.path.exists(true_label_path)):
        print("Error: Prediction and/or ground truth files not found for optimal model. Cannot generate plots.")
        return

    pred_soh = np.load(pred_label_path)
    true_soh = np.load(true_label_path)

    # --- Print Final Metrics ---
    [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_soh, true_soh)
    print("\n--- Final Optimal NN Model Performance Metrics ---")
    print(f"  Mean Squared Error (MSE): {MSE:.8f}")
    print(f"  Mean Absolute Error (MAE): {MAE:.6f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {MAPE:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {RMSE:.6f}")
    print(f"  R-squared (R2): {R2:.6f}")
    print("--------------------------------------------------\n")

    # --- Generate SOH Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(true_soh, label='Actual SOH', color='blue', linewidth=2)
    plt.plot(pred_soh, label='Estimated SOH', color='red', linestyle='--', linewidth=2)
    plt.title('Final Optimal NN Model: Estimated vs. Actual SOH', fontsize=16)
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plot_path = os.path.join(final_save_folder, 'final_optimal_nn_soh_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Final SOH comparison plot for optimal NN model saved to: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze and Plot NN Sensitivity Analysis Results and Train Optimal Model')
    parser.add_argument('--base_folder', type=str, 
                        default='optimized_plots/NN_sensitivity_analysis',
                        help='Base folder containing sensitivity analysis results')
    parser.add_argument('--final_output_folder', type=str,
                        default='optimized_plots/NN_Final_Optimized_Results',
                        help='Folder to save the final optimal model results')
    
    args = parser.parse_args()

    base_output_folder = args.base_folder
    all_sensitivity_results = []

    # Part 1: Analyze Sensitivity Results and Generate Aggregated Plots/Summaries
    for sensitivity_type_folder in os.listdir(base_output_folder):
        sensitivity_type_path = os.path.join(base_output_folder, sensitivity_type_folder)
        if os.path.isdir(sensitivity_type_path) and '_sensitivity' in sensitivity_type_folder:
            param_name = sensitivity_type_folder.replace('_sensitivity', '')
            results = analyze_and_plot_sensitivity_type(base_output_folder, sensitivity_type_folder, param_name)
            all_sensitivity_results.extend(results)

    # Part 2: Identify Optimal Hyperparameters
    if not all_sensitivity_results:
        print("No sensitivity analysis results found to determine optimal hyperparameters.")
        return

    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(all_sensitivity_results)
    
    # Find the overall optimal hyperparameters (assuming minimum loss is desired)
    # This is a simplified approach. For multi-parameter optimization, a more complex strategy
    # (e.g., grid search results, or a specific combination) would be needed.
    # For now, we'll pick the best from each individual sensitivity analysis.
    
    optimal_hyperparams = {}
    for param_name in df_results['param_name'].unique():
        best_row = df_results[df_results['param_name'] == param_name].sort_values(by='min_loss').iloc[0]
        optimal_hyperparams[param_name] = best_row['param_value']

    print("\n--- Identified Optimal Hyperparameters for NN Model ---")
    for param, value in optimal_hyperparams.items():
        print(f"  {param}: {value}")
    print("-----------------------------------------------------\n")

    # Part 3: Train Final Optimal Model and Generate Final Plots/Metrics
    train_final_optimal_nn_model(optimal_hyperparams, args.final_output_folder)

if __name__ == '__main__':
    main()