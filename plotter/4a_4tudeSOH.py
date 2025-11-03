import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scienceplots 
import matplotlib as mpl

plt.style.use(['science', 'nature'])
mpl.rcParams['text.usetex'] = False

# Set color list and inverted colormap
colors = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap', ['#377EB8', '#7BC8F6', '#FFFFFF'], N=256
)

# Set model list and MyMIT dataset
# models = ['MLP', 'CNN', 'LSTM', 'Attention-KAN-PINN']
models = ['MLP', 'NEW', 'LSTM', 'Attention-KAN-PINN']
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# Create overall figure and subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300, constrained_layout=True)

for i, model in enumerate(models):
    ax = axes[i]
    try:
        # Set file path
        root = f'../Paper_results/{data}_{model} results/Experiment1/'
        pred_label = np.load(root + 'pred_label.npy')
        true_label = np.load(root + 'true_label.npy')
    except Exception as e:
        print(f"Error loading data for {model}: {e}")
        continue

    error = np.abs(pred_label - true_label)

    # Draw scatter plot using inverted colormap
    sc = ax.scatter(true_label, pred_label, c=error, cmap=colors, s=10, alpha=0.7, vmin=0, vmax=0.1)
    ax.plot([0.65, 1.15], [0.65, 1.15], '--', c='#ff4d4e', alpha=1, linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlabel('True SOH', fontsize=14, fontweight='bold')  # Bold and set font size
    ax.set_ylabel('Prediction', fontsize=14, fontweight='bold') if i == 0 else ax.set_ylabel('')  # Keep y-axis label only for the first subplot

    # Set xlim and ylim
    ax.set_xlim(lims[data])
    ax.set_ylim(lims[data])

    # Reduce the number of y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Set to display a maximum of 5 ticks

    # Set title
    ax.set_title(f'{model}', fontsize=16, fontweight='bold')

    # Adjust tick label font size and bolding
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='inout', grid_color='black', grid_alpha=0.5)  # Bold tick lines and font

    # Bold the plot frame
    plt.setp(ax.spines.values(), linewidth=1.5)

# Add a separate color bar to the right, and adjust the width and length
cbar = fig.colorbar(
    sc, ax=axes, location='right', aspect=40, shrink=0.75, pad=0.08
)
cbar.set_label('Absolute error', fontsize=14, fontweight='bold')  # Set the color bar title to be bold and enlarged
cbar.ax.tick_params(labelsize=12, width=1.5)  # Bold the color bar ticks

# Save the figure
save_path = '4combined_figure.png'
plt.savefig(save_path, dpi=300, format='png')
print(f"Saved combined figure as {save_path}")

# Display the figure
plt.show()
