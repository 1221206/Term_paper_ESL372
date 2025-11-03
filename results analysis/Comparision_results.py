import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix  
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# Define a class to handle the results of comparative experiments
class Results:
    # Initialization function, receives a directory path and a gap value
    def __init__(self, root='../results of PINN/', gap=0.07):
        self.root = root  # Root directory to save the result files
        self.experiments = os.listdir(root)  # Get all subdirectories (experiments) under the root directory
        self.dataset = root.split('/')[-2]  # Get the dataset name
        self.gap = gap  # Threshold used to segment battery data
        self.log_dir = None  # Path to the log file
        self.pred_label = None  # Path to the prediction labels
        self.true_label = None  # Path to the true labels
        self._update_experiments(1)  # Initially, set up the first experiment

    # Function to update experiment paths, receives three parameters: training batch, test batch, and experiment number
    def _update_experiments(self, train_batch=0, test_batch=1, experiment=1):
        # Set the paths for log, prediction labels, and true labels based on the dataset name and batch numbers
        if 'XJTU' in self.dataset or 'TJU' in self.dataset:
            subfolder = f'{train_batch}-{test_batch}/Experiment{experiment}'
        else:
            subfolder = f'Experiment{experiment}'
        self.log_dir = os.path.join(self.root, subfolder, 'logging.txt')
        self.pred_label = os.path.join(self.root, subfolder, 'pred_label.npy')
        self.true_label = os.path.join(self.root, subfolder, 'true_label.npy')

    # Function to parse the log file and get the data
    def parser_log(self):
        # Create a dictionary to save the data obtained from the log file
        data_dict = {}

        # Read the log file
        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        # Parse hyperparameters, log level is CRITICAL
        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':')
                data_dict[k] = v

        # Parse the loss during training, validation, and testing
        train_data_loss = []
        valid_data_loss = []

        # Loss of the first iteration
        for i in range(len(lines)):
            line = lines[i]
            if '[train] epoch:1 iter:1 data' in line:
                train_data_loss.append(float(line.split('data loss:')[1].split('\n')[0]))

            elif '[Train]' in line:
                train_data_loss.append(float(line.split('data loss:')[1].split('\n')[0]))
            elif '[Valid]' in line:
                valid_data_loss.append(float(line.split('data loss:')[1].split('\n')[0]))

        data_dict['train_data_loss'] = train_data_loss
        data_dict['valid_data_loss'] = valid_data_loss

        # Parse data paths
        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace(f'data/{self.dataset} data/', '').replace('.csv', '').replace('\'', '').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace(f'data/{self.dataset} data/', '').replace('.csv', '').replace('\'', '').split(', ')
            for i in range(len(line_list)):
                line_list[i] = line_list[i].split('\\')[-1]
            data_dict['IDs_2'] = line_list

        # Return the dictionary containing the parsed data
        return data_dict

    # Function to parse prediction results
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
        plt.show()

        # Save the prediction results of each battery
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []
        R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff > 0.05)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))

        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            [MAE_i, MAPE_i, MSE_i, RMSE_i, R2_i] = eval_metrix(pred_i, true_i)
            start = end + 1

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

    def get_test_results(self, e):
        '''
        Parse the battery id in the training and test sets
        :param e: experiment id
        :return:
        '''
        self._update_experiments(e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict['IDs_2']

        # Check if the array lengths are consistent
        lengths = [len(v) for v in results_dict.values() if isinstance(v, list)]
        if len(set(lengths)) > 1:
            # print(f"Experiment {e} has inconsistent array lengths:", lengths)
            # Fill shorter arrays with NaN
            max_length = max(lengths)
            for key in results_dict:
                if isinstance(results_dict[key], list) and len(results_dict[key]) < max_length:
                    results_dict[key].extend([np.nan] * (max_length - len(results_dict[key])))

        # Convert to DataFrame to delete rows containing NaN
        results_df = pd.DataFrame.from_dict(results_dict)

        # Delete rows containing NaN
        results_df = results_df.dropna()

        # Convert the filtered results back to a dictionary
        return results_df.to_dict(orient='list')

    def get_battery_average(self):
        '''
        Calculate the average value of all batteries in each experiment
        :return: dataframe, including the average value of all batteries, each row represents an experiment
        '''
        df_mean_values = []
        for e in range(1, 11):
            res = self.get_test_results(e)
            df_i = pd.DataFrame(res)
            df_i = df_i[['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']]
            df_i_mean = df_i.mean(axis=0)
            df_mean_values.append(df_i_mean.values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'R2'])
        df_mean.insert(0, 'experiment', range(1, 11))
        print(df_mean)
        return df_mean

    def get_experiment_average(self):
        '''
        Get the average value of each test battery in all experiments
        :return: dataframe, each row is the average value of a battery in 10 experiments
        '''
        df_value_list = []
        for i in range(1, 11):
            res = self.get_test_results(i)
            df = pd.DataFrame(res)
            df = df[['channel', 'MAE', 'MAPE', 'MSE', 'RMSE', 'R2']]
            df = df.sort_values(by='channel')
            df.reset_index(drop=True, inplace=True)
            df_value_list.append(df[['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']].values)
        channel = df['channel']
        columns = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R2']

        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array, axis=0)
        df_mean = pd.DataFrame(np_mean, columns=columns)
        df_mean.insert(0, column='channel', value=channel)
        df_mean['channel'] = df_mean['channel']
        print(df_mean)
        return df_mean

    # Main function to test the class functionality
if __name__ == '__main__':
    root = '../My-MIT results/LSTM results/'
    writer = pd.ExcelWriter('../My-MIT results/LSTM results.xlsx')
    results = Results(root)
    df_mean1 = results.get_battery_average()
    df_mean2 = results.get_experiment_average()

    df_mean1.to_excel(writer, sheet_name='battery_mean_0', index=False)
    df_mean2.to_excel(writer, sheet_name='experiment_mean_0', index=False)
    writer.save()
    print(df_mean1.mean())
    print(df_mean2.mean())