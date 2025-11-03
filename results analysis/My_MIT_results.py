import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
plt.style.use('science')
mpl.rcParams['text.usetex'] = False

class Results:
    def __init__(self,root='C:/Users/gnana/Attention-KAN-PINN/MIT_sensitivity_analysis1') :
        self.root = root
        self.experiments = self._find_experiment_folders(root)

        self.log_dir = None
        self.pred_label = None
        self.true_label = None

    def _find_experiment_folders(self, root):
        experiment_folders = []
        for dirpath, dirnames, filenames in os.walk(root):
            if 'experiment_summary.txt' in filenames:
                experiment_folders.append(dirpath)
        return experiment_folders

    def _update_experiments(self, experiment_path):
        self.log_dir = os.path.join(experiment_path, 'logging.txt')
        self.pred_label = os.path.join(experiment_path, 'pred_label.npy')
        self.true_label = os.path.join(experiment_path, 'true_label.npy')

    def parser_log(self):
        '''
        Parse the log file generated during the training process to obtain the data
        :return: dict
        '''
        data_dict = {}

        with open(self.log_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse hyperparameters, logging level is CRITICAL
        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':',1)
                data_dict[k] = v

        # Parse the loss during the train/valid/test process
        train_data_loss = []
        train_PDE_loss = []
        train_phy_loss = []
        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []

        for i in range(len(lines)):
            line = lines[i]
            if '[train] epoch:1 iter:1 data' in line:
                parts = line.split(',')
                if len(parts) >= 5:  # Ensure parts list is long enough
                    train_data_loss.append(float(parts[1].split(':')[1]))
                    train_PDE_loss.append(float(parts[2].split(':')[1]))
                    train_phy_loss.append(float(parts[3].split(':')[1]))
                    train_total_loss.append(float(parts[4].split(':')[1]))
            elif '[Train]' in line:
                parts = line.split(',')
                if len(parts) >= 5:  # Ensure parts list is long enough
                    train_data_loss.append(float(parts[1].split(':')[1]))
                    train_PDE_loss.append(float(parts[2].split(':')[1]))
                    train_phy_loss.append(float(parts[3].split(':')[1]))
                    train_total_loss.append(float(parts[4].split(':')[1]))
            elif '[Valid]' in line:
                if 'MSE:' in line:
                    valid_data_loss.append(float(line.split('MSE:')[1].split('\n')[0]))
            elif '[Test]' in line:
                if 'MSE:' in line:
                    mse_value = line.split('MSE:')[1].split(',')[0]
                    test_mse.append(float(mse_value))
                    test_epoch.append(int(lines[i - 1].split('epoch:')[1].split(',')[0]))
                    # If 'MSE:' is not present, skip the current line to avoid IndexError
                else:
                    continue

        data_dict['train_data_loss'] = train_data_loss
        data_dict['train_PDE_loss'] = train_PDE_loss
        data_dict['train_phy_loss'] = train_phy_loss
        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch

        # Parse the data path
        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace('data/MIT data/', '').replace('.csv','').replace('\'','').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace('data/MIT data/', '').replace('.csv', '').replace('\'', '').split(', ')
            for i in range(len(line_list)):
                line_list[i] = line_list[i].split('\\')[-1]
            data_dict['IDs_2'] = line_list

        return data_dict

    def parser_label(self):
        '''
        Parse the prediction results
        :return:
        '''
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_label, true_label)
        plt.figure(figsize=(6, 4))
        plt.plot(true_label, label='true label')
        plt.plot(pred_label, label='pred label')
        plt.legend()
        #plt.show()


        # Save the prediction results of each battery
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff>0.05)[0]
        local_minima = np.concatenate((split_point,[len(true_label)]))

        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            [MAE_i, MAPE_i, MSE_i, RMSE_i, R2_i] = eval_metrix(pred_i, true_i)
            start = end+1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            MSE_list.append(MSE_i)
            RMSE_list.append(RMSE_i)
            R2_list.append(R2_i)
        results_dict = {}
        results_dict['pred_label'] = pred_label_list
        results_dict['true_label'] = true_label_list
        results_dict['MAE'] = MAE_list
        results_dict['MAPE'] = MAPE_list
        results_dict['MSE'] = MSE_list
        results_dict['RMSE'] = RMSE_list
        results_dict['R2'] = R2_list
        return results_dict

    def get_test_results(self, experiment_path):
        '''
        Parse the battery id in the training and test sets
        :param e: experiment id
        :return:
        '''
        self._update_experiments(experiment_path)
        log_dict = self.parser_log()
        results_dict = self.parser_label()

        # Check if the length of the channel is consistent, and fill it if it is inconsistent
        channel = log_dict.get('IDs_2', [])
        num_samples = len(results_dict['MAE'])  # Assume that the length of MAE is the total number of samples

        if len(channel) != num_samples:
            print(
                f"Warning: Experiment {os.path.basename(experiment_path)} - 'channel' length {len(channel)} does not match {num_samples}. Filling with None.")
            results_dict['channel'] = [None] * num_samples  # Or skip the experimental data directly
        else:
            results_dict['channel'] = channel

        return results_dict

    def get_battery_average(self):
        df_mean_values = []
        for exp_path in self.experiments:
            res = self.get_test_results(exp_path)

            # Debug: Check the length of each field
            for key, value in res.items():
                print(f"Experiment {os.path.basename(exp_path)} - {key}: length {len(value)}")

            # Make sure the fields are consistent before building the DataFrame
            try:
                df_i = pd.DataFrame(res)
                df_i = df_i[['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']]
                df_i_mean = df_i.mean(axis=0)
                df_mean_values.append(df_i_mean.values)
            except ValueError as ve:
                print(f"Error in Experiment {os.path.basename(exp_path)}: {ve}")
                continue  # Skip the problematic experiment

        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'R2'])
        df_mean.insert(0, 'experiment', [os.path.basename(p) for p in self.experiments])
        print(df_mean)
        return df_mean

    def get_experiment_average(self):
        '''
        Get the average value of each test battery in all experiments
        :return: dataframe, each row is the average value of a battery in 10 experiments
        '''
        df_value_list = []
        for exp_path in self.experiments:
            res = self.get_test_results(exp_path)
            df = pd.DataFrame(res)
            df = df[['channel', 'MAE', 'MAPE' ,'MSE', 'RMSE', 'R2']]
            df = df.sort_values(by='channel')
            df.reset_index(drop=True, inplace=True)
            df_value_list.append(df[['MAE', 'MAPE' ,'MSE', 'RMSE', 'R2']].values)
        channel = df['channel']
        columns = ['MAE', 'MAPE' ,'MSE', 'RMSE', 'R2']

        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array, axis=0)
        df_mean = pd.DataFrame(np_mean, columns=columns)
        df_mean.insert(0, column='channel', value=channel)
        df_mean['channel'] = df_mean['channel']
        print(df_mean)
        return df_mean


if __name__ == '__main__':
    root_dir = 'C:/Users/gnana/Attention-KAN-PINN/MIT_sensitivity_analysis1/'
    output_path = os.path.join(root_dir, 'analysis_results')
    os.makedirs(output_path, exist_ok=True)

    for subfolder in os.listdir(root_dir):
        full_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(full_path) and subfolder != "analysis_results":
            print(f"\nProcessing: {subfolder}")
            writer_path = os.path.join(output_path, f"{subfolder}_analysis_results.xlsx")
            
            results = Results(full_path)
            df_mean1 = results.get_battery_average()
            df_mean2 = results.get_experiment_average()

            with pd.ExcelWriter(writer_path, engine='openpyxl') as writer:
                df_mean1.to_excel(writer, sheet_name='battery_mean_0', index=False)
                df_mean2.to_excel(writer, sheet_name='experiment_mean_0', index=False)

            valid_columns = df_mean2.select_dtypes(include=[np.number]).columns
            mean_values = df_mean2[valid_columns].mean()
            print(mean_values)
    
    plt.show()


