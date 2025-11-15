
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import seaborn as sns

def main():
    features_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_features.npy'
    labels_path = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/pca_transformed_labels.npy'
    save_dir = 'Term_paper_ESL372/KANN_PINN_with_PCA/with_11_components/components_vs_soh'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features = np.load(features_path)
    labels = np.load(labels_path)

    num_components = features.shape[1]
    
    # Plot 3 components at a time. We will plot PCi, PCi+1, and SOH.
    for i in range(0, num_components - 1, 2):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        pc1_index = i
        pc2_index = i + 1
        
        if pc2_index >= num_components:
            break

        x = features[:, pc1_index]
        y = features[:, pc2_index]
        z = labels.flatten()

        scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
        
        ax.set_xlabel(f'Principal Component {pc1_index + 1}')
        ax.set_ylabel(f'Principal Component {pc2_index + 1}')
        ax.set_zlabel('SOH')
        ax.set_title(f'SOH vs. Principal Components {pc1_index + 1} & {pc2_index + 1}')
        
        fig.colorbar(scatter, ax=ax, label='SOH')
        
        plot_path = os.path.join(save_dir, f'soh_vs_pc_{pc1_index + 1}_{pc2_index + 1}.png')
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot to {plot_path}")
        plt.close()

    # --- Reality Check: Verify Independence of Principal Components ---
    print("\n--- Verifying Independence of Principal Components ---")
    
    # Create a pandas DataFrame for easy correlation calculation
    pc_df = pd.DataFrame(features, columns=[f'PC {i+1}' for i in range(num_components)])
    
    # Compute the correlation matrix
    correlation_matrix = pc_df.corr()
    
    print("\nCorrelation Matrix of Principal Components:")
    print(correlation_matrix)
    
    # Generate and save a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Principal Components')
    heatmap_path = os.path.join(save_dir, 'pca_components_correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    print(f"\nSaved correlation heatmap to {heatmap_path}")
    plt.close()

if __name__ == '__main__':
    main()
