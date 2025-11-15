
import argparse
import os
import torch
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to Python path
# This assumes the script is in Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add current dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Model_KANN_PINN_PCA import PINN_PCA

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Hybrid KAN-PINN model with PCA')
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
    parser.add_argument('--lambda3', type=float, default=1e-2, help='loss coefficient for monotonicity loss')
    parser.add_argument('--l2_lambda', type=float, default=0.0, help='L2 regularization coefficient') # Set to 0.0 as per user request
    parser.add_argument('--log_dir', type=str, default='logging_pca_sensitivity.txt', help='log dir')
    parser.add_argument('--save_folder', type=str, default='Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/sensitivity_analysis_hybrid', help='save folder')
    args = parser.parse_args()
    return args

def load_pca_data(args):
    features_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_features.npy'
    cycle_index_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/cycle_index.npy'
    labels_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_labels.npy'
    
    pca_features = np.load(features_path)
    cycle_index = np.load(cycle_index_path)
    labels = np.load(labels_path)

    # Concatenate PCA features and cycle index to form the full input 'xt'
    # The model expects xt to be (features, time)
    xt_full = np.concatenate((pca_features, cycle_index), axis=1)

    # Now, create x1, x2, y1, y2 based on xt_full and labels
    # x1 is current cycle's xt, y1 is current cycle's label
    # x2 is next cycle's xt, y2 is next cycle's label
    x1 = xt_full[:-1]
    y1 = labels[:-1]
    x2 = xt_full[1:]
    y2 = labels[1:]

    tensor_X1 = torch.from_numpy(x1).float()
    tensor_X2 = torch.from_numpy(x2).float()
    tensor_Y1 = torch.from_numpy(y1).float().view(-1, 1)
    tensor_Y2 = torch.from_numpy(y2).float().view(-1, 1)

    train_valid_X1, test_X1, train_valid_X2, test_X2, train_valid_Y1, test_Y1, train_valid_Y2, test_Y2 = \
        train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
    
    train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
        train_test_split(train_valid_X1, train_valid_X2, train_valid_Y1, train_valid_Y2, test_size=0.2, random_state=420)

    train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2), batch_size=args.batch_size, shuffle=False)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

def save_experiment_config(args, folder):
    config_path = os.path.join(folder, 'config.txt')
    with open(config_path, 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f"{k}: {v}\n")

def analyze_lambda_pca(args, base_folder, param_name, param_values):
    folder = os.path.join(base_folder, f'{param_name}_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    original_value = getattr(args, param_name)

    for value in param_values:
        exp_folder = os.path.join(folder, f'{param_name}_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        
        setattr(args, param_name, value)
        setattr(args, 'save_folder', exp_folder)
        
        print(f"\n--- Running Analysis for {param_name} = {value} ---")
        
        dataloader = load_pca_data(args)
        pinn = PINN_PCA(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
        pinn.clear_logger() # Clear logger to avoid duplicate handlers
        save_experiment_config(args, exp_folder)
    
    setattr(args, param_name, original_value) # Reset to default

def main():
    args = get_args()
    base_folder = args.save_folder
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # Define hyperparameter ranges
    lambda1_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    lambda2_values = [0.1, 0.3, 0.6, 1.0, 2.0]
    lambda3_values = [1e-3, 5e-3, 1e-2, 3e-2, 5e-2]

    # Run analysis
    analyze_lambda_pca(args, base_folder, 'lambda1', lambda1_values)
    analyze_lambda_pca(args, base_folder, 'lambda2', lambda2_values)
    analyze_lambda_pca(args, base_folder, 'lambda3', lambda3_values)
    
    print("\n--- Sensitivity Analysis Complete ---")

if __name__ == '__main__':
    main()
