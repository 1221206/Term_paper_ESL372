
import numpy as np
import pandas as pd
import os
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader.dataloader import MyMITdata
import argparse

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for feature analysis')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    # Add other necessary arguments from main_MIT.py if needed for dataloader
    parser.add_argument('--data_root', type=str, default='../MIT_data', help='Root directory for the MIT dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--log_dir', type=str, default=None, help='log dir')
    parser.add_argument('--save_folder', type=str, default=None, help='save folder')
    parser.add_argument('--output_dir', type=str, default='feature_analysis_results', help='Directory to save output plots')
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
    for x1, _, y1, _ in all_data_loader['test_3']:
        x1_list.append(x1.numpy())
        y1_list.append(y1.numpy())
        
    if not x1_list:
        print("No data loaded. Please check the data path and file structure.")
        return None, None

    features = np.concatenate(x1_list, axis=0)
    soh = np.concatenate(y1_list, axis=0)
    return features, soh

def main():
    args = get_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 1. Load data
    print("Loading data...")
    features, soh = load_all_data_as_numpy(args)
    
    if features is None:
        return
        
    print(f"Data loaded successfully. Feature matrix shape: {features.shape}, SOH shape: {soh.shape}")

    # The dataloader already normalizes the data, but for PCA it's good practice to scale it.
    # Since the dataloader uses min-max to [-1, 1], we can proceed.
    # If we were unsure, standardizing would be a safe bet:
    # features = StandardScaler().fit_transform(features)

    # 2. Perform PCA
    print("\nPerforming PCA...")
    pca = PCA()
    pca.fit(features)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    print(f"Explained variance by component: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {cumulative_explained_variance}")

    # Plot PCA explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.xticks(ticks=range(1, len(explained_variance_ratio) + 1))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title('PCA Explained Variance')
    pca_plot_path = os.path.join(args.output_dir, 'pca_explained_variance.png')
    plt.savefig(pca_plot_path)
    print(f"Saved PCA explained variance plot to {pca_plot_path}")
    plt.close()

    # 3. Perform SVD
    print("\nPerforming SVD...")
    # Center the data for SVD
    features_centered = features - features.mean(axis=0)
    U, s, Vt = np.linalg.svd(features_centered, full_matrices=False)

    print(f"Shape of U: {U.shape}")
    print(f"Shape of s (Singular values): {s.shape}")
    print(f"Shape of Vt: {Vt.shape}")
    print(f"\nSingular values (s): \n{s}")

    # Plot singular values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(s) + 1), s, 'ro-', linewidth=2)
    plt.title('Singular Values (Scree Plot)')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    svd_plot_path = os.path.join(args.output_dir, 'svd_singular_values.png')
    plt.savefig(svd_plot_path)
    print(f"Saved SVD singular values plot to {svd_plot_path}")
    plt.close()

    # 4. Correlation heatmap
    print("\nGenerating correlation heatmap...")
    feature_df = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(features.shape[1])])
    soh_df = pd.DataFrame(soh, columns=['SOH'])
    combined_df = pd.concat([feature_df, soh_df], axis=1)
    
    correlation_matrix = combined_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap between Features and SOH')
    plt.tight_layout()
    heatmap_path = os.path.join(args.output_dir, 'feature_soh_correlation_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved correlation heatmap to {heatmap_path}")
    plt.close()


if __name__ == '__main__':
    main()
