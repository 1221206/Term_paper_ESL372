import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
from scipy.stats import spearmanr

print("Current working directory:", os.getcwd())
def analyze_sensitivity_results(base_folder='MIT_sensitivity_analysis'):
    """
    Analyze the results of hyperparameter sensitivity experiments

    Args:
        base_folder: The base folder containing all sensitivity experiment results
    """
    # results dir
    results_folder = os.path.join(base_folder, 'analysis_results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # single param analysis
    analyze_single_parameter(base_folder, 'lambda1', results_folder)
    analyze_single_parameter(base_folder, 'lambda2', results_folder)
    analyze_single_parameter(base_folder, 'lambda3', results_folder)
    analyze_single_parameter(base_folder, 'l2_lambda', results_folder)

    # param combination analysis
    analyze_parameter_combination(base_folder, 'lambda1_lambda2', results_folder)
    analyze_parameter_combination(base_folder, 'lambda2_lambda3', results_folder)

    # overall sensitivity analysis
    analyze_overall_sensitivity(base_folder, results_folder)


def analyze_single_parameter(base_folder, param_name, results_folder):
    """
    Analyze the sensitivity of a single parameter

    Args:
        base_folder: Base experiment directory
        param_name: Name of the parameter
        results_folder: Folder to save analysis results
    """
    print(f"sensitivity for {param_name} ...")

    # save dir
    param_folder = os.path.join(base_folder, f'{param_name}_sensitivity')
    if not os.path.exists(param_folder):
        print(f"warning: {param_folder} doesnt exist, skiping analysis")
        return

    # results
    experiments = []
    for exp_dir in os.listdir(param_folder):
        if not exp_dir.startswith(param_name):
            continue

        # extract param val
        param_value = float(exp_dir.split('_')[-1])

        summary_path = os.path.join(param_folder, exp_dir, 'experiment_summary.txt')
        if not os.path.exists(summary_path):
            print(f"Warning: {summary_path} does not exist, skipping.")
            continue

        # read summary file
        best_metrics = {}
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'MSE:' in line and 'Best metrics' in lines[lines.index(line) - 1]:
                    best_metrics['MSE'] = float(line.split(':')[-1].strip())
                elif 'MAE:' in line:
                    best_metrics['MAE'] = float(line.split(':')[-1].strip())
                elif 'MAPE:' in line:
                    best_metrics['MAPE'] = float(line.split(':')[-1].strip())
                elif 'RMSE:' in line:
                    best_metrics['RMSE'] = float(line.split(':')[-1].strip())
                elif 'Best epoch:' in line:
                    best_metrics['Best_epoch'] = int(line.split(':')[-1].strip())

        # training loss history
        try:
            epoch_losses = np.load(os.path.join(param_folder, exp_dir, 'epoch_losses.npy'))
            valid_mses = np.load(os.path.join(param_folder, exp_dir, 'valid_mses.npy'))

            # convergence speed, error< threshold
            convergence_threshold = np.min(valid_mses) * 1.1  # 1.1 times min threshold
            convergence_epoch = np.argmax(valid_mses <= convergence_threshold) + 1

            experiments.append({
                'param_value': param_value,
                'MSE': best_metrics.get('MSE', np.nan),
                'MAE': best_metrics.get('MAE', np.nan),
                'MAPE': best_metrics.get('MAPE', np.nan),
                'RMSE': best_metrics.get('RMSE', np.nan),
                'Best_epoch': best_metrics.get('Best_epoch', np.nan),
                'convergence_epoch': convergence_epoch,
                'min_valid_mse': np.min(valid_mses),
                'final_loss': epoch_losses[-1] if len(epoch_losses) > 0 else np.nan,
                'epoch_losses': epoch_losses,
                'valid_mses': valid_mses
            })
        except:
            print(f"Warning: Failed to load loss data for {exp_dir}")

    if not experiments:
        print(f"no valud data found {param_name}")
        return

    # convert to DF and sort by param value, see how MSE, MAE changes with hyperparams
    df = pd.DataFrame(experiments)
    df = df.sort_values('param_value')

    # select metrics into csv
    csv_path = os.path.join(results_folder, f'{param_name}_sensitivity.csv')
    df[['param_value', 'MSE', 'MAE', 'MAPE', 'RMSE', 'Best_epoch', 'convergence_epoch', 'min_valid_mse',
        'final_loss']].to_csv(csv_path, index=False)

    # use spearman coeff to relate hyperparams and performance
    correlations = {}
    metrics = ['MSE', 'MAE', 'MAPE', 'RMSE', 'convergence_epoch']
    for metric in metrics:
        if all(~np.isnan(df[metric])):
            corr, p_value = spearmanr(df['param_value'], df[metric])
            correlations[metric] = (corr, p_value)

    # sensitivity score avg(mod(spearman coeff))
    # This indicates how strongly each metric depends on this parameter
    sensitivity_score = np.mean([abs(corr) for corr, _ in correlations.values()])

    # plot performance metrics vs params
    plt.figure(figsize=(15, 10))

    # MSE vs params
    plt.subplot(2, 2, 1)
    plt.plot(df['param_value'], df['MSE'], 'o-', linewidth=2)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(f'MSE vs {param_name}', fontsize=14)
    plt.grid(True)

    # MAE vs params
    plt.subplot(2, 2, 2)
    plt.plot(df['param_value'], df['MAE'], 'o-', linewidth=2)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title(f'MAE vs {param_name}', fontsize=14)
    plt.grid(True)

    # convergence speed vs params
    plt.subplot(2, 2, 3)
    plt.plot(df['param_value'], df['convergence_epoch'], 'o-', linewidth=2)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Convergence Epoch', fontsize=12)
    plt.title(f'Convergence Speed vs {param_name}', fontsize=14)
    plt.grid(True)

    # training loss curves
    plt.subplot(2, 2, 4)
    for i, row in df.iterrows():
        plt.plot(row['epoch_losses'], label=f"{param_name}={row['param_value']}")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'{param_name}_sensitivity_analysis.png'), dpi=300)
    plt.close()

    # save sensitivity analysis results
    with open(os.path.join(results_folder, f'{param_name}_sensitivity_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f" sensitivity analysis results - {param_name}\n")
        f.write(f"==============================\n")
        f.write(f"sensitivity score: {sensitivity_score:.4f}\n\n")
        f.write("spearman correlation coeff (Spearman):\n")
        for metric, (corr, p_value) in correlations.items():
            f.write(f"  {metric}: {corr:.4f} (p={p_value:.4f})\n")

        f.write("\nBest param values:\n")
        best_idx = df['MSE'].idxmin()
        f.write(f" W.R.T MSE: {param_name}={df.loc[best_idx, 'param_value']:.6f} (MSE={df.loc[best_idx, 'MSE']:.6f})\n")

        best_idx = df['MAE'].idxmin()
        f.write(f" W.R.T MAE: {param_name}={df.loc[best_idx, 'param_value']:.6f} (MAE={df.loc[best_idx, 'MAE']:.6f})\n")

        # sensitive region
        if len(df) >= 2:
            mse_changes = np.abs(np.diff(df['MSE'].values) / np.diff(df['param_value'].values))
            if len(mse_changes) > 0:
                max_change_idx = np.argmax(mse_changes)
                f.write("\n sensitive region\n")
                f.write(
                    f"  Interval with largest MSE change rate: {param_name} ∈ [{df['param_value'].iloc[max_change_idx]:.6f}, {df['param_value'].iloc[max_change_idx + 1]:.6f}]\n")
                f.write(f"  MSE change rate in the interval: {mse_changes[max_change_idx]:.6f} per unit\n")

    print(f"sensitivity analysis for {param_name} completed.")
    return df


def analyze_parameter_combination(base_folder, combo_name, results_folder):
    """
    Analyze the sensitivity of a parameter combination.

    Args:
        base_folder: Base experiment directory
        combo_name: Name of the parameter combination (e.g., 'lambda1_lambda2')
        results_folder: Directory to save the analysis results
    """
    print(f"Analyzing sensitivity of parameter combination {combo_name}...")

    param_names = combo_name.split('_')
    if len(param_names) != 2:
        print(f"warning: {combo_name} is not a valid combination")
        return

    param1, param2 = param_names

    # folder with results
    combo_folder = os.path.join(base_folder, f'{combo_name}_sensitivity')
    if not os.path.exists(combo_folder):
        print(f"warning: {combo_folder} doesnt exist, skipping analysis")
        return

    # collect results
    experiments = []
    pattern = re.compile(f"{param1}_(.+)_{param2}_(.+)")

    for exp_dir in os.listdir(combo_folder):
        match = pattern.match(exp_dir)
        if not match:
            continue

        param1_value = float(match.group(1))
        param2_value = float(match.group(2))

        # read exp summary
        summary_path = os.path.join(combo_folder, exp_dir, 'experiment_summary.txt')
        if not os.path.exists(summary_path):
            print(f"warnning: {summary_path} DNE, skiped")
            continue

        # parse summary file
        best_metrics = {}
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'MSE:' in line and 'Best metrics' in lines[lines.index(line) - 1]:
                    best_metrics['MSE'] = float(line.split(':')[-1].strip())
                elif 'MAE:' in line:
                    best_metrics['MAE'] = float(line.split(':')[-1].strip())

        experiments.append({
            'param1_value': param1_value,
            'param2_value': param2_value,
            'MSE': best_metrics.get('MSE', np.nan),
            'MAE': best_metrics.get('MAE', np.nan)
        })

    if not experiments:
        print(f"sensitivity data not found for {combo_name} ")
        return

    # convert to DF
    df = pd.DataFrame(experiments)

    # save as CSV
    csv_path = os.path.join(results_folder, f'{combo_name}_sensitivity.csv')
    df.to_csv(csv_path, index=False)

    # heatmap of MSE
    try:
        # unique values of params
        param1_values = sorted(df['param1_value'].unique())
        param2_values = sorted(df['param2_value'].unique())

        # create grid
        mse_grid = np.full((len(param1_values), len(param2_values)), np.nan)
        mae_grid = np.full((len(param1_values), len(param2_values)), np.nan)

        # fill
        for i, p1 in enumerate(param1_values):
            for j, p2 in enumerate(param2_values):
                mask = (df['param1_value'] == p1) & (df['param2_value'] == p2)
                if mask.any():
                    mse_grid[i, j] = df.loc[mask, 'MSE'].values[0]
                    mae_grid[i, j] = df.loc[mask, 'MAE'].values[0]

        # plt mse heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(mse_grid, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='MSE')
        plt.xticks(np.arange(len(param2_values)), param2_values)
        plt.yticks(np.arange(len(param1_values)), param1_values)
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.title(f'MSE Heatmap for {param1} vs {param2}')
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(mse_grid[i, j]):
                    plt.text(j, i, f"{mse_grid[i, j]:.4f}",
                             ha="center", va="center",
                             color="white" if mse_grid[i, j] > np.nanmean(mse_grid) else "black")

        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'{combo_name}_mse_heatmap.png'), dpi=300)
        plt.close()

        # fill optimal combination
        min_idx = np.nanargmin(mse_grid)
        min_i, min_j = np.unravel_index(min_idx, mse_grid.shape)
        optimal_param1 = param1_values[min_i]
        optimal_param2 = param2_values[min_j]

        # save analysis result
        with open(os.path.join(results_folder, f'{combo_name}_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write(f"parameter combination analysis result - {combo_name}\n")
            f.write(f"==============================\n")
            f.write(f"optimal combo w.r.t MSE):\n")
            f.write(f"  {param1} = {optimal_param1}\n")
            f.write(f"  {param2} = {optimal_param2}\n")
            f.write(f"  MSE = {mse_grid[min_i, min_j]:.6f}\n\n")

            f.write(f"parameter interaction analysis:\n")
            # compute parameter interaction analysis
            row_effects = np.nanmean(mse_grid, axis=1) - np.nanmean(mse_grid)
            col_effects = np.nanmean(mse_grid, axis=0) - np.nanmean(mse_grid)

            interaction = mse_grid - (
                        np.nanmean(mse_grid) + np.reshape(row_effects, (-1, 1)) + np.reshape(col_effects, (1, -1)))

            # interaction strength
            interaction_strength = np.nanstd(interaction) / np.nanstd(mse_grid)

            f.write(f"  Interaction Strength: {interaction_strength:.4f}\n")
            if interaction_strength > 0.3:
                f.write(f"  Conclusion: Strong interaction detected; parameters should not be optimized independently\n")
            elif interaction_strength > 0.1:
                f.write(f"  Conclusion: Moderate interaction detected; parameter optimization should consider combined effects\n")
            else:
                f.write(f"  Conclusion: Weak interaction; parameters can be optimized relatively independently\n")

    except Exception as e:
        print(f"Error while plotting Heatmap: {e}")

    print(f"Sensitivity analysis for parameter combination {combo_name} ")
    return df


def analyze_overall_sensitivity(base_folder, results_folder):
    """
    Comprehensive analysis of sensitivity for all parameters

    Args:
        base_folder: Base folder containing experiment data
        results_folder: Folder to save analysis results
    """

    print("start overall sensitivity analysis...")

    # Read sensitivity analysis results for each parameter
    param_names = ['lambda1', 'lambda2', 'lambda3', 'l2_lambda']
    sensitivity_scores = {}

    for param in param_names:
        analysis_file = os.path.join(results_folder, f'{param}_sensitivity_analysis.txt')
        if not os.path.exists(analysis_file):
            print(f"Warning: {analysis_file} does not exist, skipping")
            continue

        with open(analysis_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Sensitivity Score:' in line:    # "_" "_" look here
                    sensitivity_scores[param] = float(line.split(':')[-1].strip())
                    break

    if not sensitivity_scores:
        print("Not enough sensitivity analysis results found to perform comprehensive analysis, overall sensitivity.")
        return

    # sensitivity compariosn plt
    plt.figure(figsize=(10, 6))
    params = list(sensitivity_scores.keys())
    scores = [sensitivity_scores[p] for p in params]

    plt.bar(params, scores, color='steelblue')
    plt.ylabel('Sensitivity Score', fontsize=12)
    plt.title('Parameter Sensitivity Comparison', fontsize=14)
    plt.xticks(rotation=0)

    # value label
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'overall_sensitivity_comparison.png'), dpi=300)
    plt.close()

    # save overall sensitivity analysis
    with open(os.path.join(results_folder, 'overall_sensitivity_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("Hyper parameter sensitivty\n")
        f.write("============================\n\n")

        # sort params by sensitivity
        sorted_params = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)

        f.write("Parameter Sensitivity Ranking (High to Low:\n")
        for param, score in sorted_params:
            f.write(f"  {param}: {score:.4f}\n")

        f.write("\nTuning Recommendations:\n")
        f.write("1. Start by tuning the most sensitive parameters, as they have the greatest impact on model performance.\n")
        most_sensitive = sorted_params[0][0] if sorted_params else ""
        f.write(f"2. {most_sensitive} is the most sensitive parameter and should be prioritized during tuning.\n")
        f.write("3. Parameter combination analysis indicates that some parameters may have interactions, so it's recommended to consider combination analysis results during tuning.\n")
        f.write("4. For low-sensitivity parameters, you can use default values or search over a coarser grid.\n")

    print("Overall sensitivity analysis completed.")


def plot_convergence_comparison(base_folder, results_folder):
    """
    Compare convergence curves under different parameter settings

    Args:
        base_folder: Base experiment folder
        results_folder: Folder to save results
    """
    print("Plotting convergence curve comparison...")

    # find best param value
    param_names = ['lambda1', 'lambda2', 'lambda3', 'l2_lambda']
    best_settings = {}

    for param in param_names:
        csv_path = os.path.join(results_folder, f'{param}_sensitivity.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if 'MSE' in df.columns and not df['MSE'].isna().all():
            best_idx = df['MSE'].idxmin()
            best_value = df.loc[best_idx, 'param_value']
            best_settings[param] = best_value

    # find validation MSE w.r.t best params
    best_valid_curves = {}
    for param, value in best_settings.items():
        param_folder = os.path.join(base_folder, f'{param}_sensitivity')
        if not os.path.exists(param_folder):
            continue

        exp_dir = f"{param}_{value}"
        # find matching folders
        matched_dirs = [d for d in os.listdir(param_folder) if d.startswith(exp_dir)]
        if not matched_dirs:
            continue

        valid_mses_path = os.path.join(param_folder, matched_dirs[0], 'valid_mses.npy')
        if os.path.exists(valid_mses_path):
            valid_mses = np.load(valid_mses_path)
            best_valid_curves[f"{param}={value}"] = valid_mses

    # plot convergence curve
    if best_valid_curves:
        plt.figure(figsize=(12, 6))
        for label, curve in best_valid_curves.items():
            epochs = np.arange(1, len(curve) + 1)
            plt.plot(epochs, curve, label=label, linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation MSE', fontsize=12)
        plt.title('Convergence Curves for Best Parameter Settings', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # used a log scale also
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, 'best_convergence_comparison.png'), dpi=300)
        plt.close()

    print("convergence curev comparison is done")


def create_optimal_config(base_folder, results_folder):
    """
    Create optimal configuration file based on sensitivity analysis

    Args:
        base_folder: Base experiment folder
        results_folder: Folder to save results
    """
    print("Creating optimal configuration file...")

    # optimal param value
    optimal_params = {}
    param_names = ['lambda1', 'lambda2', 'lambda3', 'l2_lambda']

    for param in param_names:
        csv_path = os.path.join(results_folder, f'{param}_sensitivity.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if 'MSE' in df.columns and not df['MSE'].isna().all():
            best_idx = df['MSE'].idxmin()
            optimal_params[param] = df.loc[best_idx, 'param_value']

    # results of combined analysis
    combo_names = ['lambda1_lambda2', 'lambda2_lambda3']
    for combo in combo_names:
        analysis_file = os.path.join(results_folder, f'{combo}_analysis.txt')
        if not os.path.exists(analysis_file):
            continue

        param1, param2 = combo.split('_')
        with open(analysis_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'optimal combination' in line: # ---> "_" "_" look here
                    for j in range(1, 3):
                        if i + j < len(lines) and param1 in lines[i + j]:
                            optimal_params[param1] = float(lines[i + j].split('=')[-1].strip())
                        if i + j < len(lines) and param2 in lines[i + j]:
                            optimal_params[param2] = float(lines[i + j].split('=')[-1].strip())

    # dir
    with open(os.path.join(results_folder, 'optimal_configuration.txt'), 'w', encoding='utf-8') as f:
        f.write("Optimal configuration from hyperparameter sensitivity analysis\n")
        f.write("============================\n\n")
        f.write("Recommended settings:\n")
        for param, value in optimal_params.items():
            f.write(f"{param} = {value}\n")

    print("optimal config file created")


if __name__ == "__main__":
    # set base dir
    base_folder = 'MIT_sensitivity_analysis1'

    # sensitivity data
    analyze_sensitivity_results(base_folder)

    # plt convergence curve comparison 
    results_folder = os.path.join(base_folder, 'analysis_results')
    plot_convergence_comparison(base_folder, results_folder)

    create_optimal_config(base_folder, results_folder)

    print("Hyperparameter sensitivity analysis complete! '_'' ")
