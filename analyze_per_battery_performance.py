import argparse
import os
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import MyMITdata
from Model.Model_MLPtoKANs_for_plotting import PINN
from utils.util import eval_metrix

mpl.rcParams['text.usetex'] = False

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Final KAN-PINN model without L2 on PDE')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    # Add dummy arguments that are present in the PINN class __init__ but not needed for plotting
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epoch')
    parser.add_argument('--final_lr', type=float, default=2e-6, help='final lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate of F')
    parser.add_argument('--warmup_epochs_F', type=int, default=20, help='warmup epoch of F')
    parser.add_argument('--warmup_lr_F', type=float, default=1e-4, help='warmup lr of F')
    parser.add_argument('--final_lr_F', type=float, default=2e-6, help='final lr of F')
    parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')
    
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--alpha', type=float, default=0.9, help='loss coefficient for PDE loss')
    parser.add_argument('--beta', type=float, default=0.1, help='loss coefficient for physics loss')
    parser.add_argument('--save_folder', type=str, default='optimized_plots/Final_Model_without_L2_PDE_Results', help='save folder')
    args = parser.parse_args()
    return args

def get_test_files():
    root = r'C:\Users\pc\Inf Hyp\Power system\Pi_KANN\MIT_data'
    test_list = []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
    return test_list

def main():
    args = get_args()
    
    # --- Load Trained Model ---
    model_path = os.path.join(args.save_folder, 'model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return
        
    trainer = PINN(args)
    trainer.load_model(model_path)
    trainer.eval()
    print("Trained model loaded successfully.")

    test_files = get_test_files()
    data_loader = MyMITdata(root=r'C:\Users\pc\Inf Hyp\Power system\Pi_KANN\MIT_data', args=args)

    all_true_soh = []
    all_pred_soh = []

    for battery_file in test_files:
        print(f"Processing battery: {os.path.basename(battery_file)}")
        
        # Create a dataloader for a single battery
        single_battery_loader = data_loader.read_all(specific_path_list=[battery_file])['test_3']
        
        true_soh, pred_soh = trainer.test_for_plotting(single_battery_loader)
        
        all_true_soh.append(true_soh)
        all_pred_soh.append(pred_soh)

    all_true_soh = np.concatenate(all_true_soh)
    all_pred_soh = np.concatenate(all_pred_soh)

    # --- Calculate Metrics ---
    [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(all_pred_soh, all_true_soh)

    # --- Generate Scatter Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(all_true_soh, all_pred_soh, alpha=0.5, label='Predictions')
    
    # Plot the y=x line for reference
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
    
    ax.set_title('Predicted SOH vs. True SOH for all Test Batteries', fontsize=16)
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

    plot_path = os.path.join(args.save_folder, 'predicted_vs_true_soh_scatter_with_metrics.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Scatter plot saved to: {plot_path}")
    plt.close()

    # --- Print Metrics ---
    print("\n--- Performance Metrics for all Test Batteries ---")
    print(f"  MSE: {MSE:.8f}")
    print(f"  MAE: {MAE:.6f}")
    print(f"  MAPE: {MAPE:.6f}")
    print(f"  RMSE: {RMSE:.6f}")
    print(f"  R2: {R2:.6f}")

if __name__ == '__main__':
    main()
