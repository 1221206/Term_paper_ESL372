
import argparse
import os
import torch
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Model_KANN_PINN_PCA import PINN_PCA

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Final Hybrid KAN-PINN model with PCA')
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
    
    # Optimal hyperparameters from sensitivity analysis
    parser.add_argument('--lambda1', type=float, default=0.5, help='loss coefficient for data loss')
    parser.add_argument('--lambda2', type=float, default=2.0, help='loss coefficient for PDE loss')
    parser.add_argument('--lambda3', type=float, default=0.01, help='loss coefficient for monotonicity loss')
    
    parser.add_argument('--l2_lambda', type=float, default=0.0, help='L2 regularization coefficient') # Disabled
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir')
    parser.add_argument('--save_folder', type=str, default='Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/Final_Hybrid_KANN_PINN_with_PCA_Results', help='save folder')
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
    xt_full = np.concatenate((pca_features, cycle_index), axis=1)

    # Now, create x1, x2, y1, y2 based on xt_full and labels
    x1 = xt_full[:-1]
    y1 = labels[:-1]
    x2 = xt_full[1:]
    y2 = labels[1:]

    tensor_X1 = torch.from_numpy(x1).float()
    tensor_X2 = torch.from_numpy(x2).float()
    tensor_Y1 = torch.from_numpy(y1).float().view(-1, 1)
    tensor_Y2 = torch.from_numpy(y2).float().view(-1, 1)
    tensor_cycle_index = torch.from_numpy(cycle_index[:-1]).float().view(-1, 1) # Cycle index for x1/y1

    train_valid_X1, test_X1, train_valid_X2, test_X2, train_valid_Y1, test_Y1, train_valid_Y2, test_Y2, train_valid_cycle_index, test_cycle_index = \
        train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, tensor_cycle_index, test_size=0.2, random_state=420)
    
    train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2, train_cycle_index, valid_cycle_index = \
        train_test_split(train_valid_X1, train_valid_X2, train_valid_Y1, train_valid_Y2, train_valid_cycle_index, test_size=0.2, random_state=420)

    train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2), batch_size=args.batch_size, shuffle=False)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

def main():
    args = get_args()
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Save experiment configuration
    config_path = os.path.join(args.save_folder, 'config.txt')
    with open(config_path, 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f"{k}: {v}\n")

    print(f"Training final model with optimal hyperparameters: lambda1={args.lambda1}, lambda2={args.lambda2}, lambda3={args.lambda3}")
    
    dataloader = load_pca_data(args)

    pinn = PINN_PCA(args)
    pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
    
    print("\n--- Final Model Training Complete ---")

    # Rename config.txt to model_architecture.txt
    original_config_path = os.path.join(args.save_folder, 'config.txt')
    new_config_path = os.path.join(args.save_folder, 'model_architecture.txt')
    if os.path.exists(original_config_path):
        os.rename(original_config_path, new_config_path)
        print(f"Renamed {original_config_path} to {new_config_path}")

if __name__ == '__main__':
    main()
