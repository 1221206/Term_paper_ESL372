import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
import sys
import seaborn as sns

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import MyMITdata
from Model.Model_NN_PINN import PINN as Trainer
from utils.util import eval_metrix

def get_args_for_trainer():
    parser = argparse.ArgumentParser('Hyper Parameters for NN-PINN model')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate for F')
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--lambda1', type=float, default=1.0, help='lambda1 for loss')
    parser.add_argument('--lambda2', type=float, default=1.0, help='lambda2 for loss')
    parser.add_argument('--lambda3', type=float, default=1.0, help='lambda3 for loss')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='optimized_plots/NN_PINN_Final_Optimized_Results', help='save folder')
    return parser.parse_args([])

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
    losses_path = os.path.join(experiment_folder, 'epoch_losses.npy')
    if not os.path.exists(losses_path):
        print(f"Error: epoch_losses.npy not found in {experiment_folder}")
        return
    losses = np.load(losses_path)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Total Loss', color='blue')
    plt.title(f'NN-PINN Sensitivity: {param_name} = {param_value} (Total Loss)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plot_path = os.path.join(experiment_folder, f'{param_name}_{param_value}_total_loss.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

def analyze_and_plot_sensitivity_type(base_output_folder, sensitivity_type_folder, param_name):
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
                    param_results.append({'param_name': param_name, 'param_value': param_value, 'min_loss': min_loss})
                    print(f"  {param_name} = {param_value}: Min Loss = {min_loss:.6f}")
                    plot_single_experiment_losses(experiment_path, param_name, param_value)
    if not param_results:
        return []
    param_values = [res['param_value'] for res in param_results]
    min_losses = [res['min_loss'] for res in param_results]
    sorted_indices = np.argsort(param_values)
    param_values = np.array(param_values)[sorted_indices]
    min_losses = np.array(min_losses)[sorted_indices]
    analysis_results_folder = os.path.join(base_output_folder, 'analysis_results')
    os.makedirs(analysis_results_folder, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, min_losses, marker='o', linestyle='-', color='green')
    plt.title(f'NN-PINN Sensitivity Analysis: Min Total Loss vs. {param_name}', fontsize=16)
    plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Minimum Total Loss', fontsize=12)
    if 'lambda' in param_name:
        plt.xscale('log')
    plt.grid(True)
    aggregated_plot_path = os.path.join(analysis_results_folder, f'{param_name}_sensitivity_analysis.png')
    plt.savefig(aggregated_plot_path, dpi=300)
    plt.close()
    summary_df = pd.DataFrame({param_name: param_values, 'Min_Total_Loss': min_losses})
    csv_path = os.path.join(analysis_results_folder, f'{param_name}_sensitivity.csv')
    summary_df.to_csv(csv_path, index=False)
    return param_results

def analyze_and_plot_2d_sensitivity(base_output_folder, sensitivity_type_folder, param_names):
    sensitivity_type_path = os.path.join(base_output_folder, sensitivity_type_folder)
    results = []
    print(f"Analyzing {param_names[0]} and {param_names[1]} sensitivity...")
    for experiment_folder_name in sorted(os.listdir(sensitivity_type_path)):
        experiment_path = os.path.join(sensitivity_type_path, experiment_folder_name)
        if os.path.isdir(experiment_path):
            parts = experiment_folder_name.replace(f'{param_names[0]}_', '').replace(f'_{param_names[1]}_', ' ').split()
            try:
                param1_val, param2_val = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            losses_path = os.path.join(experiment_path, 'epoch_losses.npy')
            if os.path.exists(losses_path):
                losses = np.load(losses_path)
                if len(losses) > 0:
                    min_loss = np.min(losses)
                    results.append({param_names[0]: param1_val, param_names[1]: param2_val, 'min_loss': min_loss})
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index=param_names[0], columns=param_names[1], values='min_loss')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis")
    plt.title(f'NN-PINN: Min Total Loss vs. {param_names[0]} and {param_names[1]}')
    analysis_results_folder = os.path.join(base_output_folder, 'analysis_results')
    os.makedirs(analysis_results_folder, exist_ok=True)
    heatmap_path = os.path.join(analysis_results_folder, f'{param_names[0]}_{param_names[1]}_mse_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    csv_path = os.path.join(analysis_results_folder, f'{param_names[0]}_{param_names[1]}_sensitivity.csv')
    df.to_csv(csv_path, index=False)
    return df

def train_final_optimal_nn_pinn_model(optimal_hyperparams, final_save_folder):
    print("\n--- Starting Final NN-PINN Model Training with Optimal Hyperparameters ---")
    os.makedirs(final_save_folder, exist_ok=True)
    args = get_args_for_trainer()
    for param, value in optimal_hyperparams.items():
        setattr(args, param, value)
    setattr(args, 'save_folder', final_save_folder)
    setattr(args, 'log_dir', 'logging.txt')
    dataloader = load_MyMIT_data(args)
    trainer = Trainer(args)
    trainer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
    print("--- Final NN-PINN Model Training Finished ---")
    pred_label_path = os.path.join(final_save_folder, 'pred_label.npy')
    true_label_path = os.path.join(final_save_folder, 'true_label.npy')
    if not (os.path.exists(pred_label_path) and os.path.exists(true_label_path)):
        return
    pred_soh, true_soh = np.load(pred_label_path), np.load(true_label_path)
    [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_soh, true_soh)
    print("\n--- Final Optimal NN-PINN Model Performance Metrics ---")
    print(f"  MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}, R2: {R2:.6f}")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(true_soh, label='Actual SOH', color='blue', linewidth=2)
    plt.plot(pred_soh, label='Estimated SOH', color='red', linestyle='--', linewidth=2)
    plt.title('Final Optimal NN-PINN Model: Estimated vs. Actual SOH', fontsize=16)
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plot_path = os.path.join(final_save_folder, 'final_optimal_nn_pinn_soh_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze NN-PINN Sensitivity and Train Optimal Model')
    parser.add_argument('--base_folder', type=str, default='optimized_plots/NN_PINN_sensitivity_analysis', help='Base folder for sensitivity results')
    parser.add_argument('--final_output_folder', type=str, default='optimized_plots/NN_PINN_Final_Optimized_Results', help='Folder for final optimal model results')
    args = parser.parse_args()
    base_output_folder = args.base_folder
    all_sensitivity_results = []
    all_2d_dfs = {}

    for sensitivity_type_folder in os.listdir(base_output_folder):
        path = os.path.join(base_output_folder, sensitivity_type_folder)
        if os.path.isdir(path) and '_sensitivity' in sensitivity_type_folder:
            if 'lambda1_lambda2' in sensitivity_type_folder:
                df = analyze_and_plot_2d_sensitivity(base_output_folder, sensitivity_type_folder, ['lambda1', 'lambda2'])
                if not df.empty: all_2d_dfs['lambda1_lambda2'] = df
            elif 'lambda2_lambda3' in sensitivity_type_folder:
                df = analyze_and_plot_2d_sensitivity(base_output_folder, sensitivity_type_folder, ['lambda2', 'lambda3'])
                if not df.empty: all_2d_dfs['lambda2_lambda3'] = df
            else:
                param_name = sensitivity_type_folder.replace('_sensitivity', '')
                results = analyze_and_plot_sensitivity_type(base_output_folder, sensitivity_type_folder, param_name)
                all_sensitivity_results.extend(results)

    if not all_sensitivity_results and not all_2d_dfs:
        print("No sensitivity analysis results found.")
        return

    optimal_hyperparams = {}
    if all_sensitivity_results:
        df_results = pd.DataFrame(all_sensitivity_results)
        for param_name in df_results['param_name'].unique():
            best_row = df_results[df_results['param_name'] == param_name].sort_values(by='min_loss').iloc[0]
            optimal_hyperparams[param_name] = best_row['param_value']
    
    if 'lambda1_lambda2' in all_2d_dfs:
        best_l1_l2 = all_2d_dfs['lambda1_lambda2'].sort_values(by='min_loss').iloc[0]
        optimal_hyperparams['lambda1'] = best_l1_l2['lambda1']
        optimal_hyperparams['lambda2'] = best_l1_l2['lambda2']

    if 'lambda2_lambda3' in all_2d_dfs:
        best_l2_l3 = all_2d_dfs['lambda2_lambda3'].sort_values(by='min_loss').iloc[0]
        if 'lambda2' not in optimal_hyperparams or best_l2_l3['min_loss'] < best_l1_l2['min_loss']:
            optimal_hyperparams['lambda2'] = best_l2_l3['lambda2']
        optimal_hyperparams['lambda3'] = best_l2_l3['lambda3']

    print("\n--- Identified Optimal Hyperparameters for NN-PINN Model ---")
    for param, value in optimal_hyperparams.items():
        print(f"  {param}: {value}")
    print("-----------------------------------------------------\\n")

    train_final_optimal_nn_pinn_model(optimal_hyperparams, args.final_output_folder)

if __name__ == '__main__':
    main()
