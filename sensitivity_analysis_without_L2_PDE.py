
from dataloader.dataloader import MyMITdata
from Model.Model_MLPtoKANs_without_L2_PDE import PINN
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for MIT dataset')
    parser.add_argument('--data', type=str, default='MyMIT', help='MyMIT')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=25, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup lr')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-6, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-5, help='learning rate of F')
    parser.add_argument('--warmup_epochs_F', type=int, default=20, help='warmup epoch of F')
    parser.add_argument('--warmup_lr_F', type=float, default=1e-4, help='warmup lr of F')
    parser.add_argument('--final_lr_F', type=float, default=2e-6, help='final lr of F')


    # model related
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='MIT results', help='save folder')

    args = parser.parse_args()

    return args

def load_MyMIT_data(args):
    root = '..\MIT_data'
    train_list = []
    test_list = []
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root,f))
            else:
                train_list.append(os.path.join(batch_root,f))
    data = MyMITdata(root=root,args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader
def main():
    args = get_args()

    # result directory
    base_folder = 'MIT_sensitivity_analysis_without_L2_PDE'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # single param sensitivity analysis - alpha
    analyze_alpha(args, base_folder)

    #- beta
    analyze_beta(args, base_folder)

    # combined paramter sensitivity analysis- alpha beta
    analyze_alpha_beta(args, base_folder)


def analyze_alpha(args, base_folder):
    """analyze sensitivity of alpha"""
    folder = os.path.join(base_folder, 'alpha_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # test
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for value in alpha_values:
        # subdirectory
        exp_folder = os.path.join(folder, f'alpha_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        # modify params
        setattr(args, 'alpha', value)
        setattr(args, 'save_folder', exp_folder)

        # load data and train model
        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

        save_experiment_config(args, exp_folder)


def analyze_beta(args, base_folder):
    """sensitivity of beta"""
    folder = os.path.join(base_folder, 'beta_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # test
    beta_values = [0.05, 0.1, 0.2, 0.3, 0.5]

    for value in beta_values:
        # sub dir
        exp_folder = os.path.join(folder, f'beta_{value}')
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        setattr(args, 'beta', value)
        setattr(args, 'save_folder', exp_folder)

        dataloader = load_MyMIT_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

        save_experiment_config(args, exp_folder)


def analyze_alpha_beta(args, base_folder):
    """sensitivity of alpha and beta combined"""
    folder = os.path.join(base_folder, 'alpha_beta_sensitivity')
    if not os.path.exists(folder):
        os.makedirs(folder)

    alpha_values = [0.5, 0.7, 0.9]
    beta_values = [0.1, 0.2, 0.3]

    for a in alpha_values:
        for b in beta_values:
            # sub dir
            exp_folder = os.path.join(folder, f'alpha_{a}_beta_{b}')
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)

            setattr(args, 'alpha', a)
            setattr(args, 'beta', b)
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

if __name__ == '__main__':
    #pass
    main()
