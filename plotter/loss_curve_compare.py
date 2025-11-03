import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import ticker

# create file Final_Paper_results

plt.style.use(['science', 'nature'])
from matplotlib.backends.backend_pdf import PdfPages

# Set color list and inverted colormap
colors = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap', ['#377EB8', '#7BC8F6', '#FFFFFF'], N=256
)

# Set model names and MyMIT dataset
models = ['MLP', 'CNN', 'LSTM', 'KAN', 'KAN-PINN', 'Attention-KAN', 'Attention-KAN-PINN(without PDE)', 'Attention-KAN-PINN']
data = 'MyMIT'
lims = {'MyMIT': [0.79, 1.005]}

# Generate a new plot for all models
fig, ax = plt.subplots(figsize=(7, 5), dpi=600)

# Create subplot: zoom in on the small loss value area in the bottom right corner of the main plot
ax_inset = fig.add_axes([0.35, 0.20, 0.35, 0.35])  # Zoomed-in subplot position and size
# Reduce the number of y-axis ticks by half
ax_inset.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower', nbins=3))
# Iterate through model names and load their respective loss data
for model in models:
    try:
        # Set file path
        root = f'../Final_Paper_results/{data}_{model} results1/Experiment10/'

        # Set file path
        root = f'../Final_Paper_results/{data}_{model} results1/Experiment10/'

        # Assume the loss values are stored in the epoch_losses.npy file, and the loss for each epoch is an array
        loss_file = root + 'epoch_losses.npy'
        loss = np.load(loss_file, allow_pickle=True)  # Assume the file is epoch_losses.npy
        print(f"Loaded loss data from {loss_file} for {model}.")

        # Print basic information of the loss data for debugging
        print(f"Total epochs: {len(loss)}")

        # Check if there is data
        if len(loss) == 0:
            raise ValueError(f"No loss data found in {loss_file}")

        epochs = np.arange(1, len(loss) + 1)  # Generate Epochs

        # If no loss data is loaded, skip the current model
        if len(loss) == 0:
            print(f"No loss data for {model}. Skipping...")
            continue

        # Filter the loss values, keeping only those less than or equal to 0.020
        filtered_epochs = epochs[loss <= 0.020]
        filtered_loss = loss[loss <= 0.020]

        # Plot the loss curve for the main plot
        if model == 'MLP':
            ax.plot(filtered_epochs, filtered_loss, color='blue', linestyle='--', marker='s', markersize=2.5, alpha=0.7,
                    label=model)
        elif model == 'CNN':
            ax.plot(filtered_epochs, filtered_loss, color='green', linestyle='-.', marker='^', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'LSTM':
            ax.plot(filtered_epochs, filtered_loss, color='red', linestyle=':', marker='D', markersize=2.5, alpha=0.7,
                    label=model)
        elif model == 'KAN':
            ax.plot(filtered_epochs, filtered_loss, color='orange', linestyle='-', marker='x', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'KAN-PINN':
            ax.plot(filtered_epochs, filtered_loss, color='magenta', linestyle='-.', marker='p', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'Attention-KAN':
            ax.plot(filtered_epochs, filtered_loss, color='brown', linestyle='--', marker='*', markersize=2.5,
                    alpha=0.7, label=model)
        elif model == 'Attention-KAN-PINN(without PDE)':
            ax.plot(filtered_epochs, filtered_loss, color='teal', linestyle=':', marker='h', markersize=2.5, alpha=0.7,
                    label=model)
        elif model == 'Attention-KAN-PINN':
            ax.plot(filtered_epochs, filtered_loss, color='purple', linestyle='-', marker='o', markersize=2.5,
                    alpha=0.7, label=model)

        # Filter the part of the loss value that is less than 0.001
        small_loss_mask = loss <= 0.001
        small_epochs = epochs[small_loss_mask]
        small_loss = loss[small_loss_mask]

        # Plot the small loss value part on the subplot, using the same color and style as the main plot
        if model == 'MLP':
            ax_inset.plot(small_epochs, small_loss, color='blue', linestyle='--', marker='s', markersize=2.5, alpha=0.7)
        elif model == 'CNN':
            ax_inset.plot(small_epochs, small_loss, color='green', linestyle='-.', marker='^', markersize=2.5,
                          alpha=0.7)
        elif model == 'LSTM':
            ax_inset.plot(small_epochs, small_loss, color='red', linestyle=':', marker='D', markersize=2.5, alpha=0.7)
        elif model == 'KAN':
            ax_inset.plot(small_epochs, small_loss, color='orange', linestyle='-', marker='x', markersize=2.5,
                          alpha=0.7)
        elif model == 'KAN-PINN':
            ax_inset.plot(small_epochs, small_loss, color='magenta', linestyle='-.', marker='p', markersize=2.5,
                          alpha=0.7)
        elif model == 'Attention-KAN':
            ax_inset.plot(small_epochs, small_loss, color='brown', linestyle='--', marker='*', markersize=2.5,
                          alpha=0.7)
        elif model == 'Attention-KAN-PINN(without PDE)':
            ax_inset.plot(small_epochs, small_loss, color='teal', linestyle=':', marker='h', markersize=2.5, alpha=0.7)
        elif model == 'Attention-KAN-PINN':
            ax_inset.plot(small_epochs, small_loss, color='purple', linestyle='-', marker='o', markersize=2.5,
                          alpha=0.7)

    except Exception as e:
        print(f"Error loading data for {model}: {e}")

# Set the x and y axis labels for the main plot
ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')  # Training Epochs
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')  # Loss value

# Adjust tick label font size and bolding
ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='inout', grid_color='black',
               grid_alpha=0.5)  # Bold tick lines and font

# Add grid
ax.grid(True, linestyle='--', alpha=0.6)

# Get the legend handles and labels of the main plot
handles, labels = ax.get_legend_handles_labels()

# Place the legend above the subplot, arranged in two columns
ax_inset.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.7, 1.28),  # Control the position of the legend directly above the subplot
    ncol=2,                      # Display in two columns
    fontsize=12,
    frameon=False
)

# Bold the plot frame
plt.setp(ax.spines.values(), linewidth=1.5)

# Set the grid for the subplot
ax_inset.grid(True, linestyle='--', alpha=0.6)
ax_inset.tick_params(axis='both', which='major', labelsize=10, width=2, length=6, direction='inout', grid_color='black',
                     grid_alpha=0.5)
# Use `constrained_layout` instead of `tight_layout`
plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Manually adjust the spacing to avoid warnings

# Save the loss curve image
plt.savefig('loss_curve_comparison_expanded_models.png', dpi=600, bbox_inches='tight')

# Display the figure
plt.show()