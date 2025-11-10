import argparse
import os
import torch
import numpy as np
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import MyMITdata
from Model.KANN_PINN_without_monotonic import PINN
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import matplotlib as mpl # Keep mpl import for other rcParams if needed

# No scienceplots, no usetex
# plt.style.use('science')
# mpl.rcParams['text.usetex'] = False
# plt.rcParams['text.usetex'] = False
mpl.rcParams['text.usetex'] = False # Ensure LaTeX is disabled

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Final KAN-PINN model without monotonicity loss')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate of F')
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u') # Default from sensitivity_analysis_KAN_PINN.py
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u') # Default from sensitivity_analysis_KAN_PINN.py
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F') # Default from sensitivity_analysis_KAN_PINN.py
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F') # Default from sensitivity_analysis_KAN_PINN.py
    parser.add_argument('--lambda1', type=float, default=0.1, help='loss coefficient for data loss') # Optimal from analysis
    parser.add_argument('--lambda2', type=float, default=0.1, help='loss coefficient for PDE loss') # Optimal from analysis
    parser.add_argument('--lambda3', type=float, default=0.0, help='loss coefficient for physics loss (set to 0 for no monotonicity)') # Optimal from analysis
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization coefficient') # Default from sensitivity_analysis_KAN_PINN.py
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='optimized_plots/Final_KANN_PINN_without_monotonic_Results', help='save folder')
    args = parser.parse_args()
    return args

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

def main():
    args = get_args()
    
    # Create save folder if it doesn't exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    dataloader = load_MyMIT_data(args)
    trainer = PINN(args)
    trainer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

    # --- Load Results and Generate Plots ---
    print("\n--- Generating Final Plots and Metrics for Final KAN-PINN Model ---")
    pred_label_path = os.path.join(args.save_folder, 'pred_label.npy')
    true_label_path = os.path.join(args.save_folder, 'true_label.npy')

    if not (os.path.exists(pred_label_path) and os.path.exists(true_label_path)):
        print("Error: Prediction and/or ground truth files not found for optimal model. Cannot generate plots.")
        return

    pred_soh = np.load(pred_label_path)
    true_soh = np.load(true_label_path)

    # --- Print Final Metrics ---
    [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_soh, true_soh)
    print("\n--- Final Optimal KAN-PINN Model Performance Metrics ---")
    print(f"  MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}, R2: {R2:.6f}")

    # --- Generate SOH Plot ---
    # plt.style.use('science') # Removed this line
    plt.figure(figsize=(12, 7))
    plt.plot(true_soh, label='Actual SOH', color='blue', linewidth=2)
    plt.plot(pred_soh, label='Estimated SOH', color='red', linestyle='--', linewidth=2)
    plt.title('Final Optimal KAN-PINN Model: Estimated vs. Actual SOH', fontsize=16)
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plot_path = os.path.join(args.save_folder, 'final_optimal_kan_pinn_soh_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Final SOH comparison plot for optimal KAN-PINN model saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
