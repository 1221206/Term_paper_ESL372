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
from Model.Model_MLPtoKANs_for_old_plotting import PINN
from utils.util import eval_metrix

mpl.rcParams['text.usetex'] = False

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Old KAN-PINN model analysis')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    
    # Arguments from logging.txt for the old model
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.0001, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-05, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-06, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-05, help='learning rate of F')
    
    parser.add_argument('--u_layers_num', type=int, default=4, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    
    parser.add_argument('--lambda1', type=float, default=5.0, help='loss coefficient for data loss')
    parser.add_argument('--lambda2', type=float, default=0.6, help='loss coefficient for PDE loss')
    parser.add_argument('--lambda3', type=float, default=0.01, help='loss coefficient for physics loss')
    parser.add_argument('--l2_lambda', type=float, default=1e-05, help='L2 regularization coefficient')

    # These are not used for loading the model, but PINN expects them
    parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default=None, help='save folder') # Not used for saving model in this script

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
    model_path = r'C:\Users\pc\Inf Hyp\Power system\Pi_KANN\Term_paper_ESL372\MIT_sensitivity_analysis1\lambda1_sensitivity\lambda1_5.0\model.pth'
    
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
        
        true_soh, pred_soh, _ = trainer.test_for_plotting(single_battery_loader) # _ to ignore metrics per battery
        
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
    
    ax.set_title('Predicted SOH vs. True SOH for Old Model (lambda1=5.0)', fontsize=16)
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

    output_folder = r'C:\Users\pc\Inf Hyp\Power system\Pi_KANN\Term_paper_ESL372\MIT_sensitivity_analysis1\lambda1_sensitivity\lambda1_5.0\plots'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_path = os.path.join(output_folder, 'predicted_vs_true_soh_scatter_with_metrics.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Scatter plot saved to: {plot_path}")
    plt.close()

    # --- Print Metrics ---
    print("\n--- Performance Metrics for Old Model (lambda1=5.0) ---")
    print(f"  MSE: {MSE:.8f}")
    print(f"  MAE: {MAE:.6f}")
    print(f"  MAPE: {MAPE:.6f}")
    print(f"  RMSE: {RMSE:.6f}")
    print(f"  R2: {R2:.6f}")

if __name__ == '__main__':
    main()
