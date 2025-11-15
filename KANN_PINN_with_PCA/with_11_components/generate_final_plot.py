import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from Model_KANN_PINN_PCA import PINN_PCA # Assuming the model class is in this file
import argparse

def load_full_dataset():
    features_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_features.npy'
    labels_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_labels.npy'
    cycle_index_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/cycle_index.npy'
    
    pca_features = np.load(features_path)
    labels = np.load(labels_path)
    cycle_index = np.load(cycle_index_path)

    # Create x1, x2, y1, y2
    xt = np.concatenate((pca_features, cycle_index.reshape(-1, 1)), axis=1)
    x1 = xt[:-1]
    x2 = xt[1:]
    y1 = labels[:-1]
    y2 = labels[1:]

    tensor_X1 = torch.from_numpy(x1).float()
    tensor_X2 = torch.from_numpy(x2).float()
    
    return tensor_X1, tensor_X2, y1, cycle_index[:-1]

def main():
    results_dir = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/Final_Hybrid_KANN_PINN_with_PCA_Results'
    plots_dir = os.path.join(results_dir, 'plots')
    model_path = os.path.join(results_dir, 'model.pth')

    # --- Determine Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    # Need to create a dummy args object for model initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--u_layers_num', type=int, default=3)
    parser.add_argument('--u_hidden_dim', type=int, default=60)
    parser.add_argument('--F_layers_num', type=int, default=3)
    parser.add_argument('--F_hidden_dim', type=int, default=60)
    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=2.0)
    parser.add_argument('--lambda3', type=float, default=0.01)
    parser.add_argument('--l2_lambda', type=float, default=0.0)
    parser.add_argument('--warmup_lr', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--lr_F', type=float, default=1e-05)
    parser.add_argument('--epochs', type=int, default=200) # Added epochs
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_folder', type=str, default=None)
    args = parser.parse_args([])

    model = PINN_PCA(args).to(device) # Move model to device
    saved_model = torch.load(model_path, map_location=device) # Ensure loading on correct device
    model.solution_u.load_state_dict(saved_model['solution_u'])
    model.dynamical_F.load_state_dict(saved_model['dynamical_F'])
    model.eval()

    # --- Get Predictions for Entire Dataset ---
    full_X1, full_X2, full_Y1, full_cycle_index = load_full_dataset()
    full_X1 = full_X1.to(device) # Move data to device
    
    with torch.no_grad():
        full_pred_Y1 = model.solution_u(full_X1)
    
    full_pred_Y1 = full_pred_Y1.cpu().numpy().flatten() # Move back to CPU for numpy/plotting
    full_Y1 = full_Y1.flatten()
    full_cycle_index = full_cycle_index.flatten()

    # --- Plot 1: Full Time Series Plot ---
    # Calculate metrics for the time series plot
    mse_ts = mean_squared_error(full_Y1, full_pred_Y1)
    mae_ts = mean_absolute_error(full_Y1, full_pred_Y1)
    r2_ts = r2_score(full_Y1, full_pred_Y1)
    mape_ts = np.mean(np.abs((full_Y1 - full_pred_Y1) / np.where(full_Y1 == 0, 1, full_Y1))) * 100
    rmse_ts = np.sqrt(mse_ts)

    plt.style.use('seaborn-v0_8-whitegrid') # Apply seaborn style
    plt.figure(figsize=(12, 7)) # Use figsize from reference script

    # Plot against continuous index
    continuous_index = np.arange(len(full_Y1))
    plt.plot(continuous_index, full_Y1, label='True SOH', color='blue', linewidth=2) # Use linewidth from reference
    plt.plot(continuous_index, full_pred_Y1, label='Predicted SOH', color='red', linestyle='--', linewidth=2) # Use linewidth from reference
    plt.title('Final Optimal Hybrid KANN-PINN with PCA: SOH vs. Cycle Number (Full Dataset)', fontsize=16) # Use fontsize from reference
    plt.xlabel('Cycle Number', fontsize=12) # Use fontsize from reference
    plt.ylabel('SOH', fontsize=12) # Use fontsize from reference
    plt.legend(fontsize=10) # Use fontsize from reference
    plt.grid(True)

    # Add metrics to the plot
    metrics_text_ts = f'MSE: {mse_ts:.6f}\nMAE: {mae_ts:.6f}\nRMSE: {rmse_ts:.6f}\nR2: {r2_ts:.4f}\nMAPE: {mape_ts:.4f}%'
    plt.text(0.05, 0.95, metrics_text_ts, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    time_series_plot_path = os.path.join(plots_dir, 'soh_time_series_final.png')
    plt.savefig(time_series_plot_path, dpi=300)
    print(f"Final SOH time series plot saved to {time_series_plot_path}")
    plt.close()

    # --- Plot 1b: Zoomed-in Time Series Plot ---
    zoom_slice = int(len(continuous_index) * 0.2)
    plt.style.use('seaborn-v0_8-whitegrid') # Apply seaborn style
    plt.figure(figsize=(12, 7)) # Use figsize from reference script
    plt.plot(continuous_index[:zoom_slice], full_Y1[:zoom_slice], label='True SOH', color='blue', linewidth=2) # Use linewidth from reference
    plt.plot(continuous_index[:zoom_slice], full_pred_Y1[:zoom_slice], label='Predicted SOH', color='red', linestyle='--', linewidth=2) # Use linewidth from reference
    plt.title('Zoomed-in SOH vs. Cycle Number (First 20% of Full Dataset)', fontsize=16) # Use fontsize from reference
    plt.xlabel('Cycle Number', fontsize=12) # Use fontsize from reference
    plt.ylabel('SOH', fontsize=12) # Use fontsize from reference
    plt.legend(fontsize=10) # Use fontsize from reference
    plt.grid(True)

    zoomed_plot_path = os.path.join(plots_dir, 'soh_time_series_zoomed.png')
    plt.savefig(zoomed_plot_path, dpi=300)
    print(f"Zoomed-in SOH time series plot saved to {zoomed_plot_path}")
    plt.close()

    # --- Plot 2: Scatter Plot (using test set data) ---
    true_label_path = os.path.join(results_dir, 'true_label.npy')
    pred_label_path = os.path.join(results_dir, 'pred_label.npy')
    true_label = np.load(true_label_path).flatten()
    pred_label = np.load(pred_label_path).flatten()

    mse = mean_squared_error(true_label, pred_label)
    mae = mean_absolute_error(true_label, pred_label)
    r2 = r2_score(true_label, pred_label)
    mape = np.mean(np.abs((true_label - pred_label) / np.where(true_label == 0, 1, true_label))) * 100
    rmse = np.sqrt(mse)

    plt.figure(figsize=(8, 8))
    plt.scatter(true_label, pred_label, alpha=0.6, s=10)
    plt.plot([min(true_label), max(true_label)], [min(true_label), max(true_label)], 'r--', lw=2, label='Ideal Prediction')
    plt.title('Final Optimal Hybrid KANN-PINN with PCA: Predicted vs. True SOH (Test Set)')
    plt.xlabel('True SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(True)
    plt.legend()

    metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}\nR2: {r2:.4f}\nMAPE: {mape:.4f}%'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    scatter_plot_path = os.path.join(results_dir, 'final_optimal_hybrid_kann_pinn_with_pca_scatter_with_metrics.png')
    plt.savefig(scatter_plot_path, dpi=300)
    print(f"Final comparison scatter plot saved to {scatter_plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
