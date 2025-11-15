
import argparse
import os
import torch
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Model_KANN_PINN_PCA import PINN_PCA
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Final KAN-PINN model with PCA')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate of F')
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--lambda1', type=float, default=0.1, help='loss coefficient for data loss')
    parser.add_argument('--lambda2', type=float, default=0.1, help='loss coefficient for PDE loss')
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--log_dir', type=str, default='logging_pca.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/results', help='save folder')
    args = parser.parse_args()
    return args

def load_pca_data(args):
    features_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_features.npy'
    labels_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_labels.npy'
    
    features = np.load(features_path)
    labels = np.load(labels_path)

    # The dataloader in the original project creates x1, x2, y1, y2. We need to replicate that.
    # x1 is the features of the current cycle, y1 is the label of the current cycle
    # x2 is the features of the next cycle, y2 is the label of the next cycle
    # We can approximate this by shifting the data. However, this should be done per-battery,
    # and we don't have the battery boundaries here.
    # For simplicity, we will create a simple mapping of x -> y, and will need to adjust the model
    # not to expect x1, x2, y1, y2.
    
    # Let's adjust the training to be simpler: x1, y1. We will need to modify the train_one_epoch
    # in the model accordingly.
    
    # For now, let's stick to the original structure as much as possible.
    # We will assume that the data is continuous and create the shifted data.
    x1 = features[:-1]
    y1 = labels[:-1]
    x2 = features[1:]
    y2 = labels[1:]

    # Convert to tensors
    tensor_X1 = torch.from_numpy(x1).float()
    tensor_X2 = torch.from_numpy(x2).float()
    tensor_Y1 = torch.from_numpy(y1).float().view(-1, 1)
    tensor_Y2 = torch.from_numpy(y2).float().view(-1, 1)

    # Split data
    train_valid_X1, test_X1, train_valid_X2, test_X2, train_valid_Y1, test_Y1, train_valid_Y2, test_Y2 = \
        train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
    
    train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
        train_test_split(train_valid_X1, train_valid_X2, train_valid_Y1, train_valid_Y2, test_size=0.2, random_state=420)

    train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2), batch_size=args.batch_size, shuffle=False)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

def main():
    args = get_args()
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    dataloader = load_pca_data(args)
    trainer = PINN_PCA(args)
    trainer.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

    # --- Load Results and Generate Plots ---
    print("\n--- Generating Final Plots and Metrics for KAN-PINN Model with PCA ---")
    pred_label_path = os.path.join(args.save_folder, 'pred_label.npy')
    true_label_path = os.path.join(args.save_folder, 'true_label.npy')

    if not (os.path.exists(pred_label_path) and os.path.exists(true_label_path)):
        print("Error: Prediction and/or ground truth files not found. Cannot generate plots.")
        return

    pred_soh = np.load(pred_label_path)
    true_soh = np.load(true_label_path)

    # --- Print Final Metrics ---
    [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_soh, true_soh)
    print("\n--- Final KAN-PINN with PCA Model Performance Metrics ---")
    print(f"  MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}, R2: {R2:.6f}")

    # --- Generate SOH Plot ---
    plt.figure(figsize=(12, 7))
    plt.plot(true_soh, label='Actual SOH', color='blue', linewidth=2)
    plt.plot(pred_soh, label='Estimated SOH', color='red', linestyle='--', linewidth=2)
    plt.title('Final KAN-PINN with PCA Model: Estimated vs. Actual SOH', fontsize=16)
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('State of Health (SOH)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plot_path = os.path.join(args.save_folder, 'final_kan_pinn_pca_soh_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Final SOH comparison plot for KAN-PINN with PCA model saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
