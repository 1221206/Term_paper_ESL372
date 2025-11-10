import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots
import matplotlib as mpl

plt.style.use(['science', 'nature'])
mpl.rcParams['text.usetex'] = False

models = ['NN', 'KANN', 'NN_PINN', 'KAN_PINN']
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

for model in models:
    try:
        # Construct the absolute path to the results directory
        root = os.path.join(script_dir, '..', 'optimized_plots', f'{model}_Final_Optimized_Results')
        
        pred_label = np.load(os.path.join(root, 'pred_label.npy'))
        true_label = np.load(os.path.join(root, 'true_label.npy'))

        # Create a new figure for each model
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

        # Calculate absolute error
        error = np.abs(pred_label - true_label)

        # Draw scatter plot
        sc = ax.scatter(true_label, pred_label, c=error, cmap='viridis', s=10, alpha=0.7, vmin=0, vmax=0.1)
        ax.plot([0.65, 1.15], [0.65, 1.15], '--', c='#ff4d4e', alpha=1, linewidth=1)
        ax.set_aspect('equal')

        # Label settings
        ax.set_xlabel('True SOH', fontsize=14, fontweight='bold')
        ax.set_ylabel('Prediction', fontsize=14, fontweight='bold')
        ax.set_title(f'{model}', fontsize=15, fontweight='bold')

        ax.set_xlim(lims[data])
        ax.set_ylim(lims[data])

        # Tick style
        ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='inout')

        # Bold plot frame
        plt.setp(ax.spines.values(), linewidth=1.5)

        # Add color bar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Absolute error', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12, width=1.5)

        # Save image
        save_path = f'{model}_comparison.png'
        plt.savefig(save_path, dpi=300, format='png')
        print(f"Saved individual figure as {save_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error processing model {model}: {e}")
