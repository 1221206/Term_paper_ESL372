import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.util import eval_metrix

mpl.rcParams['text.usetex'] = False

def plot_results(directory):
    pred_path = os.path.join(directory, 'pred_label.npy')
    true_path = os.path.join(directory, 'true_label.npy')

    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        print(f"Skipping {directory}: pred_label.npy or true_label.npy not found.")
        return

    print(f"Processing {directory}...")

    pred_soh = np.load(pred_path)
    true_soh = np.load(true_path)

    # --- Calculate Metrics ---
    [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_soh, true_soh)

    # --- Generate Scatter Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(true_soh, pred_soh, alpha=0.5, label='Predictions')
    
    # Plot the y=x line for reference
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
    
    model_name = os.path.basename(directory).replace('_Results', '').replace('_', ' ')
    ax.set_title(f'Predicted SOH vs. True SOH for {model_name}', fontsize=16)
    ax.set_xlabel('True SOH', fontsize=12)
    ax.set_ylabel('Predicted SOH', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Add metrics to the plot
    metrics_text = (
        f"MSE: {MSE:.8f}\n"
        f"MAE: {MAE:.6f}\n"
        f"MAPE: {MAPE:.6f}\n"
        f"RMSE: {RMSE:.6f}\n"
        f"R2: {R2:.6f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_folder = os.path.join(directory, 'plots')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plot_path = os.path.join(output_folder, f'{os.path.basename(directory)}_scatter_with_metrics.png')
    plt.savefig(plot_path, dpi=300)
    print(f"  - Scatter plot saved to: {plot_path}")
    plt.close()

def main():
    base_dir = 'optimized_plots'
    target_dirs = [
        'Final_KANN_PINN_without_monotonic_Results',
        'Final_Model_without_L2_PDE_Results',
        'KAN_PINN_Final_Optimized_Results',
        'KANN_Final_Optimized_Results',
        'NN_Final_Optimized_Results',
        'NN_PINN_Final_Optimized_Results'
    ]

    for dir_name in target_dirs:
        full_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(full_path):
            plot_results(full_path)
        else:
            print(f"Directory not found: {full_path}")

if __name__ == '__main__':
    main()
