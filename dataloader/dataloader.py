import pandas as pd 
import numpy as np  
import torch  
from torch.utils.data import TensorDataset  # for tensor
from torch.utils.data import DataLoader  # for batching data
import os  # Import the os library for file and directory operations
import random  
from sklearn.model_selection import train_test_split  #train_test_split
from utils.util import write_to_txt 


class DF():  
    def __init__(self, args):  # Receives parameters
        self.normalization = True  # Set normalization to True
        self.normalization_method = args.normalization_method  # Get normalization method (min-max or z-score)
        self.args = args  # Store parameters

    def _3_sigma(self, Ser1):  # For outliers beyond 3-sigma from the mean
        '''
        Calculate the indices of values that exceed three standard deviations.
        :param Ser1: Input sequence (e.g., a column from a DataFrame).
        :return: Indices of values outside the 3-sigma range.
        '''
        # If an outlier is found, return its index
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]  
        return index  

    def delete_3_sigma(self, df):  # Remove outliers and infinities
        '''
        :param df: Input DataFrame.
        :return: DataFrame after removing outliers.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinity with NaN
        df = df.dropna()  # Drop rows with NaN
        df = df.reset_index(drop=True)  # Reset indices
        out_index = []  # Store outlier indices
        for col in df.columns:  # Get outlier indices for each column
            index = self._3_sigma(df[col])  
            out_index.extend(index)  
        out_index = list(set(out_index))  # Remove duplicate indices
        df = df.drop(out_index, axis=0)  # Drop rows with outliers
        df = df.reset_index(drop=True)  # Reset the index
        return df  

    def read_one_csv(self, file_name, nominal_capacity=None):  
        '''
        :param file_name: File name (string).
        :param nominal_capacity: Nominal capacity (optional).
        :return: DataFrame.
        '''
        df = pd.read_csv(file_name)  
        df.insert(df.shape[1] - 1, 'cycle index', np.arange(df.shape[0]))  # Insert cycle index column

        df = self.delete_3_sigma(df)  # Remove outliers

        if nominal_capacity is not None:  
            df['capacity'] = df['capacity'] / nominal_capacity  # Calculate relative capacity
            f_df = df.iloc[:, :-1]  # Get all columns except the last one
            if self.normalization_method == 'min-max':  
                f_df = 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1  # Normalize to [-1, 1]
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean()) / f_df.std()  # z-score normalization

            df.iloc[:, :-1] = f_df  # Update the DataFrame

        return df  

    def load_one_battery(self, path, nominal_capacity=None):  
        '''
        :param path: File path.
        :param nominal_capacity: Nominal capacity (optional).
        :return: Tuple of ((x1, y1), (x2, y2)).
        '''
        df = self.read_one_csv(path, nominal_capacity)  
        x = df.iloc[:, :-1].values  # Feature data
        y = df.iloc[:, -1].values  # Label data
        x1 = x[:-1]  # First n-1 features
        x2 = x[1:]  # Last n-1 features
        y1 = y[:-1]  # First n-1 labels
        y2 = y[1:]  # Last n-1 labels
        return (x1, y1), (x2, y2)  

    def load_all_battery(self, path_list, nominal_capacity):  
        '''
        :param path_list: List of file paths.
        :param nominal_capacity: Nominal capacity used for SOH calculation.
        :return: Dataloader.
        '''
        X1, X2, Y1, Y2 = [], [], [], []  # Store features and labels
        if hasattr(self.args, 'log_dir') and self.args.log_dir is not None and \
           hasattr(self.args, 'save_folder') and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)  # Create save path
            write_to_txt(save_name, 'data path:')  # Write header to the log file
            write_to_txt(save_name, str(path_list))  # Log file paths

        for path in path_list:  # For each file path
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)  # Load data
            X1.append(x1)  # Append features to X1
            X2.append(x2)  # Append features to X2
            Y1.append(y1)  # Append labels to Y1
            Y2.append(y2)  # Append labels to Y2

        # Concatenate into single arrays
        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        # Convert to tensors
        tensor_X1 = torch.from_numpy(X1).float()  
        tensor_X2 = torch.from_numpy(X2).float() 
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1, 1)  # Reshape labels to (n, 1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1, 1)  

        # Split data for training, validation, and testing
        # First, split into 80% train+valid and 20% test
        train_valid_X1, test_X1, train_valid_X2, test_X2, train_valid_Y1, test_Y1, train_valid_Y2, test_Y2 = \
            train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
        
        # Then, split train+valid into 80% train and 20% valid
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_valid_X1, train_valid_X2, train_valid_Y1, train_valid_Y2, test_size=0.2, random_state=420)

        
        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size,  
                                  shuffle=True)  
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size,  
                                  shuffle=True)  
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.args.batch_size,  
                                 shuffle=False)  

        # Use the entire dataset for full analysis
        full_dataset_loader = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                 batch_size=self.args.batch_size,  
                                 shuffle=False) 

        # Return all data loaders
        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'full_dataset': full_dataset_loader}
        return loader  


class MyMITdata(DF):  # For MIT data
    def __init__(self, root='../Dataset_plot/My data/MIT data', args=None): 
        super(MyMITdata, self).__init__(args)  # Call parent DF class
        self.root = root  
        self.batchs = ['2017-05-12', '2017-06-30', '2018-04-12']  
        if self.normalization:  # If normalization is enabled
            self.nominal_capacity = 1.1  # Nominal capacity
        else:
            self.nominal_capacity = None  

    def read_one_batch(self, batch):  
        '''
        Read all CSV files in one MIT batch.
        :param batch: int, must be in [1, 2, 3].
        :return: dict.
        '''
        assert batch in [1, 2, 3], 'batch must be in {}'.format([1, 2, 3])  
        root = os.path.join(self.root, self.batchs[batch - 1])  # Get batch location
        file_list = os.listdir(root)  # List all files in the path
        path_list = []  # Store file paths
        for file in file_list:  
            path = os.path.join(root, file)  
            path_list.append(path)  
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)  # Load battery data

    def read_all(self, specific_path_list=None):  
        '''
        Read all CSV files. If specific_path_list is provided, read only the specified files; 
        otherwise, read all available files.
        :param specific_path_list: List of file paths.
        :return: dict.
        '''
        if specific_path_list is None:  # If no path is provided, read all files
            file_list = []  
            for batch in self.batchs:  
                root = os.path.join(self.root, batch) 
                files = os.listdir(root)  
                for file in files: 
                    path = os.path.join(root, file)  
                    file_list.append(path)  
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)  
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)  


class MyTJUdata(DF):
    def __init__(self,root='Dataset_plot/My data/TJU data',args=None):
        super(MyTJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]

    def read_one_batch(self,batch):
        '''
        Read a batch of CSV files.
        :param batch: int, optional [1, 2, 3].
        :return: DataFrame.
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        df = pd.DataFrame()
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacities[batch])

    def read_all(self,specific_path_list):
        '''
        Read all CSV files and encapsulate them into a dataloader.
        :param self:
        :return: dict.
        '''
        # Determine nominal capacity based on the batch name
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)


if __name__ == '__main__':  
    import argparse  
    def get_args():  
        parser = argparse.ArgumentParser()  
        parser.add_argument('--data', type=str, default='MyMIT', help='Dataset name: TJU or MyMIT')  # Get dataset name
        parser.add_argument('--batch', type=int, default=1, help='Batch number: 1, 2, 3')  # Batch number
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size')  # Batch size
        parser.add_argument('--normalization_method', type=str, default='min-max', help='Normalization method')  # Normalization method
        parser.add_argument('--log_dir', type=str, default='test.txt', help='Log file path')  # Save file path
        return parser.parse_args()  

    args = get_args()  


    mit = MyMITdata(args=args)  # Instance for MIT
    mit.read_one_batch(batch=1)  
    loader = mit.read_all()  

    train_loader = loader['train']  
    test_loader = loader['test']  
    valid_loader = loader['valid'] 
    all_loader = loader['test_3']  
    print('train_loader:', len(train_loader), 'test_loader:', len(test_loader), 'valid_loader:', len(valid_loader), 'all_loader:', len(all_loader))  # Number of batches in each

    for iter, (x1, x2, y1, y2) in enumerate(train_loader):  # Shape and stats of the first batch
        print('x1 shape:', x1.shape)  
        print('x2 shape:', x2.shape)  
        print('y1 shape:', y1.shape)  
        print('y2 shape:', y2.shape)  
        print('y1 max:', y1.max())  
        break  
