
import numpy as np
import pandas as pd
import os
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataloader.dataloader import MyMITdata

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for feature analysis')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    parser.add_argument('--data_root', type=str, default='MIT_data', help='Root directory for the MIT dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--log_dir', type=str, default=None, help='log dir')
    parser.add_argument('--save_folder', type=str, default=None, help='save folder')
    parser.add_argument('--output_dir', type=str, default='feature_analysis_results', help='Directory to save output plots')
    parser.add_argument('--pca_model_path', type=str, default='Term_paper_ESL32/KANN_PINN_with_PCA/with_11_components/pca_model.joblib', help='Path to save the PCA model')
    args = parser.parse_args()
    return args

def load_all_data_as_numpy(args):
    """
    Loads all MIT data from all batches into a single numpy array.
    """
    root = args.data_root
    if not os.path.exists(root):
        print(f"Error: The specified data root '{root}' does not exist.")
        return None, None

    all_files = []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        if not os.path.exists(batch_root):
            continue
        files = os.listdir(batch_root)
        for f in files:
            all_files.append(os.path.join(batch_root, f))

    # Use the dataloader to read and process the files
    data_loader = MyMITdata(root=root, args=args)
    
    # Get all data from the dataloader
    all_data_loader = data_loader.read_all(specific_path_list=all_files)
    
    # Extract tensors and convert to numpy
    x1_list = []
    y1_list = []
    for x1, _, y1, _ in all_data_loader['full_dataset']:
        x1_list.append(x1.numpy())
        y1_list.append(y1.numpy())
        
    if not x1_list:
        print("No data loaded. Please check the data path and file structure.")
        return None, None

    features = np.concatenate(x1_list, axis=0)
    soh = np.concatenate(y1_list, axis=0)
    return features, soh

def load_all_data_as_numpy(args):
    """
    Loads all MIT data from all batches into a single numpy array, separating features and cycle index.
    """
    root = args.data_root
    if not os.path.exists(root):
        print(f"Error: The specified data root '{root}' does not exist.")
        return None, None, None

    all_files = []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        if not os.path.exists(batch_root):
            continue
        files = os.listdir(batch_root)
        for f in files:
            all_files.append(os.path.join(batch_root, f))

    data_loader = MyMITdata(root=root, args=args)
    all_data_loader = data_loader.read_all(specific_path_list=all_files)
    
    raw_features_list = []
    cycle_index_list = []
    soh_list = []

    for x1, _, y1, _ in all_data_loader['full_dataset']:
        # x1 contains original features + cycle index. We need to separate them.
        # Assuming cycle index is the last column of x1 as per DF.read_one_csv
        raw_features_list.append(x1[:, :-1].numpy()) # All but the last column are features
        cycle_index_list.append(x1[:, -1:].numpy()) # The last column is cycle index
        soh_list.append(y1.numpy())
        
    if not raw_features_list:
        print("No data loaded. Please check the data path and file structure.")
        return None, None, None

    raw_features = np.concatenate(raw_features_list, axis=0)
    cycle_index = np.concatenate(cycle_index_list, axis=0)
    soh = np.concatenate(soh_list, axis=0)
    
    return raw_features, cycle_index, soh

def main():
    args = get_args()
    
    # 1. Load data
    print("Loading data...")
    raw_features, cycle_index, soh = load_all_data_as_numpy(args)
    
    if raw_features is None:
        return
        
    print(f"Raw features shape: {raw_features.shape}, Cycle index shape: {cycle_index.shape}, SOH shape: {soh.shape}")

    # 2. Perform PCA on raw features only
    print("\nPerforming PCA with n_components=11 on raw features...")
    pca = PCA(n_components=11)
    pca_transformed_features = pca.fit_transform(raw_features)
    
    # 3. Save the PCA model
    pca_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pca_model.joblib')
    print(f"\nSaving PCA model to {pca_model_path}...")
    joblib.dump(pca, pca_model_path)
    print("PCA model saved successfully.")

    # 4. Save the transformed features, cycle index, and labels
    transformed_features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pca_transformed_features.npy')
    cycle_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cycle_index.npy')
    labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pca_transformed_labels.npy')
    
    print(f"\nSaving PCA transformed features to {transformed_features_path}...")
    np.save(transformed_features_path, pca_transformed_features)
    print("PCA transformed features saved successfully.")
    
    print(f"\nSaving cycle index to {cycle_index_path}...")
    np.save(cycle_index_path, cycle_index)
    print("Cycle index saved successfully.")

    print(f"\nSaving labels to {labels_path}...")
    np.save(labels_path, soh)
    print("Labels saved successfully.")

if __name__ == '__main__':
    main()
