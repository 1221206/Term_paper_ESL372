import torch
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import MyMITdata
from Model.Model_MLPtoKANs import PINN
from utils.util import eval_metrix

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Final Model Training')
    # Add all necessary arguments from main_MIT.py, but we will override the key ones.
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-6, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate of F')
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    
    # --- Optimal Hyperparameters ---
    parser.add_argument('--lambda1', type=float, default=5.0)
    parser.add_argument('--lambda2', type=float, default=2.0)
    parser.add_argument('--lambda3', type=float, default=0.01)
    parser.add_argument('--l2_lambda', type=float, default=1e-05)
    parser.add_argument('--save_folder', type=str, default='optimized_plots/Final_Optimized_Results', help='save folder')

    args = parser.parse_args()
    return args

def load_MyMIT_data(args):
    # Use an absolute path to the dataset
    root = r'C:\Users\pc\Inf Hyp\Power system\Pi_KANN\MIT_data'
    train_list = []
    test_list = []
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root,f))
            else:
                train_list.append(os.path.join(batch_root,f))
    data = MyMITdata(root=root,args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}
    return dataloader

def main():
    args = get_args()

    # --- 1. Train the Final Model ---
    print("--- Starting Final Model Training with Optimal Hyperparameters ---")
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    dataloader = load_MyMIT_data(args)
    pinn = PINN(args)
    pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
    print("--- Final Model Training Finished ---")

    # --- 2. Load Results and Generate Plots ---
    print("--- Generating Final Plots and Metrics ---")
    pred_label_path = os.path.join(args.save_folder, 'pred_label.npy')
    true_label_path = os.path.join(args.save_folder, 'true_label.npy')

    if not (os.path.exists(pred_label_path) and os.path.exists(true_label_path)):
        print("Error: Prediction and/or ground truth files not found. Cannot generate plots.")
        return

    pred_soh = np.load(pred_label_path)
    true_soh = np.load(true_label_path)

    # --- Print Final Metrics ---
    #include R2 w.r.t eval metrics
    [MAE, MAPE, MSE, RMSE,R2] = eval_metrix(pred_soh, true_soh)
    print("\n--- Final Model Performance Metrics ---")
    print(f"  Mean Squared Error (MSE): {MSE:.8f}")
    print(f"  Mean Absolute Error (MAE): {MAE:.6f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {MAPE:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {RMSE:.6f}")
    print(f"  R-squared (R2): {R2:.6f}")
    print("---------------------------------------\n")

    # --- Generate SOH Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(true_soh, label='Actual SOH', color='blue', linewidth=2)
    plt.plot(pred_soh, label='Estimated SOH', color='red', linestyle='--', linewidth=2)
    plt.title('Final Model: Estimated vs. Actual SOH', fontsize=16)
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plot_path = os.path.join(args.save_folder, 'final_soh_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Final SOH comparison plot saved to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    main()