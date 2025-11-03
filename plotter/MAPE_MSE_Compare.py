'''
Compare the results of three models on the MIT dataset: 'MAE', 'MAPE', 'MSE', 'RMSE', 'R2'
'''
import pandas as pd
import matplotlib.pyplot as plt

# Read data
KAN_MIT_results = pd.read_excel('../MIT results/MIT-KAN results.xlsx')
KAN_MIT_results1 = pd.read_excel('../MIT results/MIT-KAN results1.xlsx')
KAN_MIT_results2 = pd.read_excel('../MIT results/MIT-KAN results2.xlsx')
# Assume each file has the same column names
metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
colors = ['blue', 'orange', 'green']   # Colors for different files

for metric in metrics:
    plt.figure(figsize=(10, 5))

    plt.plot(KAN_MIT_results['experiment'], KAN_MIT_results[metric], color=colors[0], label='KAN_MIT_results')
    plt.plot(KAN_MIT_results1['experiment'], KAN_MIT_results1[metric], color=colors[1], label='KAN_MIT_results1')
    plt.plot(KAN_MIT_results2['experiment'], KAN_MIT_results2[metric], color=colors[2], label='KAN_MIT_results2')

    plt.title(f'{metric} Comparison')
    plt.xlabel('experiment')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    # plt.savefig(f'{metric}_comparison.png')  # Save the image
    plt.show()
