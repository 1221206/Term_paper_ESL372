import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots
import matplotlib as mpl

plt.style.use(['science', 'nature'])
mpl.rcParams['text.usetex'] = False

# Set color list
colors = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap', ['#377EB8', '#7BC8F6', '#FFFFFF'], N=256
)

# Extended model list
models = ['NN', 'KANN', 'NN_PINN', 'KAN_PINN']
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# Create subplots (4)
fig, axes = plt.subplots(1, len(models), figsize=(16, 5), dpi=300, constrained_layout=True)
sc = None

for i, model in enumerate(models):
    ax = axes[i]
    try:
        # Set file path
        root = f'../optimized_plots/{model}_Final_Optimized_Results/'
        pred_label = np.load(root + 'pred_label.npy')
        true_label = np.load(root + 'true_label.npy')
    except Exception as e:
        print(f"Error loading data for {model}: {e}")
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', fontsize=12)
        ax.set_xlabel('True SOH', fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Prediction', fontsize=14, fontweight='bold')
        ax.set_title(f'{model}', fontsize=15, fontweight='bold')
        ax.set_xlim(lims[data])
        ax.set_ylim(lims[data])
        continue

    # Calculate absolute error
    error = np.abs(pred_label - true_label)

    # Draw scatter plot
    sc = ax.scatter(true_label, pred_label, c=error, cmap=colors, s=10, alpha=0.7, vmin=0, vmax=0.1)
    ax.plot([0.65, 1.15], [0.65, 1.15], '--', c='#ff4d4e', alpha=1, linewidth=1)
    ax.set_aspect('equal')

    # Label settings
    ax.set_xlabel('True SOH', fontsize=14, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Prediction', fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel('')

    ax.set_xlim(lims[data])
    ax.set_ylim(lims[data])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(f'{model}', fontsize=15, fontweight='bold')

    # Tick style
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='inout')

    # Bold plot frame
    plt.setp(ax.spines.values(), linewidth=1.5)

# Add unified color bar
if sc:
    cbar = fig.colorbar(sc, ax=axes, location='right', aspect=40, shrink=0.75, pad=0.08)
    cbar.set_label('Absolute error', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12, width=1.5)

# Save image
save_path = 'all_models_comparison.png'
plt.savefig(save_path, dpi=300, format='png')
print(f"Saved combined figure as {save_path}")

plt.show()
