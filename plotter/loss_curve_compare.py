import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl # Moved to top
from matplotlib import ticker
import os # Import os module

# create file Final_Paper_results

plt.style.use(['science', 'nature'])
mpl.rcParams['text.usetex'] = False # Now mpl is defined
from matplotlib.backends.backend_pdf import PdfPages

# Set color list and inverted colormap
colors = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap', ['#377EB8', '#7BC8F6', '#FFFFFF'], N=256
)

# Set model names and MyMIT dataset
models = ['NN', 'KANN', 'NN_PINN', 'KAN_PINN'] # Updated model list
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate two new plots
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Define colors and markers for our models
model_styles = {
    'NN': {'color': 'blue', 'linestyle': '--'},
    'KANN': {'color': 'green', 'linestyle': '-.'},
    'NN_PINN': {'color': 'red', 'linestyle': ':'},
    'KAN_PINN': {'color': 'purple', 'linestyle': '-'},
}

# Plot NN and KANN on the first figure
for i, model_name in enumerate(['NN', 'KANN']):
    ax = axes1[i]
    try:
        root = os.path.join(script_dir, '..', 'optimized_plots', f'{model_name}_Final_Optimized_Results')
        loss_file = os.path.join(root, 'epoch_losses.npy')
        loss = np.load(loss_file, allow_pickle=True)
        epochs = np.arange(1, len(loss) + 1)
        style = model_styles[model_name]
        ax.plot(epochs, loss, color=style['color'], linestyle=style['linestyle'], label=model_name)
        ax.set_yscale('log')
        ax.set_title(f'{model_name} Loss Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.grid(True, which="both", ls="--")
        ax.legend()
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")

fig1.tight_layout()

# Plot NN-PINN and KAN-PINN on the second figure
for i, model_name in enumerate(['NN_PINN', 'KAN_PINN']):
    ax = axes2[i]
    try:
        root = os.path.join(script_dir, '..', 'optimized_plots', f'{model_name}_Final_Optimized_Results')
        loss_file = os.path.join(root, 'epoch_losses.npy')
        loss = np.load(loss_file, allow_pickle=True)
        epochs = np.arange(1, len(loss) + 1)
        style = model_styles[model_name]
        ax.plot(epochs, loss, color=style['color'], linestyle=style['linestyle'], label=model_name)
        ax.set_yscale('log')
        ax.set_title(f'{model_name} Loss Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.grid(True, which="both", ls="--")
        ax.legend()
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")

fig2.tight_layout()

# Display the figures
plt.show()