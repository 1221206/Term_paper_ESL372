import argparse
import os
from dataloader.dataloader import MyMITdata
from Model.Model_KAN_PINN import PINN

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for KAN-PINN model')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate of F')
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--lambda1', type=float, default=1, help='loss coefficient for data loss')
    parser.add_argument('--lambda2', type=float, default=0.6, help='loss coefficient for PDE loss')
    parser.add_argument('--lambda3', type=float, default=1e-2, help='loss coefficient for physics loss')
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='optimized_plots/KAN_PINN_sensitivity_analysis', help='save folder')
    args = parser.parse_args()
    return args

def load_MyMIT_data(args):
    root = r'C:\Users\pc\Inf Hyp\Power system\Pi_KANN\MIT_data'
    train_list, test_list = [], []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    data = MyMITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': trainloader['train_2'], 'valid': trainloader['valid_2'], 'test': testloader['test_3']}
    return dataloader

def analyze_lambda1(args, base_folder):
    folder = os.path.join(base_folder, 'lambda1_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)
    lambda1_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    for value in lambda1_values:
        exp_folder = os.path.join(folder, f'lambda1_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        setattr(args, 'lambda1', value)
        setattr(args, 'save_folder', exp_folder)
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
        save_experiment_config(args, exp_folder)

def analyze_lambda2(args, base_folder):
    folder = os.path.join(base_folder, 'lambda2_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)
    lambda2_values = [0.1, 0.3, 0.6, 1.0, 2.0]
    for value in lambda2_values:
        exp_folder = os.path.join(folder, f'lambda2_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        setattr(args, 'lambda2', value)
        setattr(args, 'save_folder', exp_folder)
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
        save_experiment_config(args, exp_folder)

def analyze_lambda3(args, base_folder):
    folder = os.path.join(base_folder, 'lambda3_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)
    lambda3_values = [1e-3, 5e-3, 1e-2, 3e-2, 5e-2]
    for value in lambda3_values:
        exp_folder = os.path.join(folder, f'lambda3_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        setattr(args, 'lambda3', value)
        setattr(args, 'save_folder', exp_folder)
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])
        save_experiment_config(args, exp_folder)

def save_experiment_config(args, folder):
    config_path = os.path.join(folder, 'config.txt')
    with open(config_path, 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f"{k}: {v}\n")

def main():
    args = get_args()
    base_folder = 'optimized_plots/KAN_PINN_sensitivity_analysis'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    analyze_lambda1(args, base_folder)
    analyze_lambda2(args, base_folder)
    analyze_lambda3(args, base_folder)

if __name__ == '__main__':
    main()
