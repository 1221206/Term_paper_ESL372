import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import AverageMeter, get_logger, eval_metrix
import os
from Model.Model_MLPtoKANs import KAN, Solution_u

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PINN(nn.Module):
    def __init__(self, args):
        super(PINN, self).__init__()
        self.args = args

        if args.save_folder and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.solution_u = Solution_u(u_hidden_dim=args.u_hidden_dim, u_layers_num=args.u_layers_num).to(device)
        self.dynamical_F = KAN(input_dim=35, output_dim=1, hidden_dim=args.F_hidden_dim,
                               num_hidden_layers=args.F_layers_num, dropout=0.2).to(device)

        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=args.epochs)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = 0 # No monotonicity loss
        self.l2_lambda = args.l2_lambda
        self.best_model = None

    def _save_args(self):
        if self.args.log_dir:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.info(f"\t{k}:{v}")

    def forward(self, xt):
        xt = xt.detach().requires_grad_(True)
        x, t = xt[:, :-1], xt[:, -1:]
        u = self.solution_u(torch.cat((x, t), dim=1))
        u_t = grad(u.sum(), t, create_graph=True, only_inputs=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, only_inputs=True)[0]
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        return u, u_t - F

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter, loss2_meter, loss3_meter, l2_reg_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for idx, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)

            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
            loss2 = 0.5 * self.loss_func(f1, torch.zeros_like(f1)) + 0.5 * self.loss_func(f2, torch.zeros_like(f2))
            loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()

            l2_reg = torch.tensor(0.).to(device)
            for param in self.solution_u.parameters():
                l2_reg += torch.norm(param)
            for param in self.dynamical_F.parameters():
                l2_reg += torch.norm(param)

            loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            l2_reg_meter.update(l2_reg.item())

        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, l2_reg_meter.avg

    def save_model_architecture(self, metrics):
        architecture_path = os.path.join(self.args.save_folder, 'model_architecture.txt')
        with open(architecture_path, 'w') as f:
            f.write("Model: Final KAN-PINN (without monotonicity loss)\n\n")
            f.write("Solution_u Network (KAN Encoder + MLP Predictor):\n")
            f.write("  Encoder (KAN):\n")
            f.write("    Input dimension: 17\n")
            f.write("    Output dimension: 32\n")
            f.write(f"    Number of hidden layers: {self.args.u_layers_num}\n")
            f.write(f"    Number of neurons in each hidden layer: {self.args.u_hidden_dim}\n")
            f.write("  Predictor (MLP):\n")
            f.write("    Input dimension: 32\n")
            f.write("    Hidden dimension: 32\n")
            f.write("    Output dimension: 1\n\n")
            f.write("Dynamical_F Network (KAN):\n")
            f.write("  Input dimension: 35\n")
            f.write("  Output dimension: 1\n")
            f.write(f"  Number of hidden layers: {self.args.F_layers_num}\n")
            f.write(f"  Number of neurons in each hidden layer: {self.args.F_hidden_dim}\n\n")
            f.write("--- Final Model Performance Metrics ---\n")
            f.write(f"  Mean Squared Error (MSE): {metrics[2]:.8f}\n")
            f.write(f"  Mean Absolute Error (MAE): {metrics[0]:.6f}\n")
            f.write(f"  Mean Absolute Percentage Error (MAPE): {metrics[1]:.6f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): {metrics[3]:.6f}\n")
            f.write(f"  R-squared (R2): {metrics[4]:.6f}\n")
            f.write("---------------------------------------\n")

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = 10
        early_stop = 0
        epoch_losses = []

        for e in range(1, self.args.epochs + 1):
            early_stop += 1
            loss1, loss2, loss3, l2_reg = self.train_one_epoch(e, trainloader)
            self.scheduler.step()
            total_loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 + self.l2_lambda * l2_reg
            epoch_losses.append(total_loss)

            info = f'[Train] epoch:{e}, lr:{self.optimizer1.param_groups[0]["lr"]:.6f}, total loss:{total_loss:.6f}'
            self.logger.info(info)

            if e > 1 and total_loss > epoch_losses[-2]:
                self.logger.info("Loss is increasing, stopping training.")
                break

            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = f'[Valid] epoch:{e}, MSE: {valid_mse}'
                self.logger.info(info)

            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_label, true_label)
                info = f'[Test] MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}, R2: {R2:.6f}'
                self.logger.info(info)
                early_stop = 0
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                    self.save_model_architecture([MAE, MAPE, MSE, RMSE, R2])

            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = f'early stop at epoch {e}'
                self.logger.info(info)
                break

        if self.args.save_folder is not None:
            np.save(os.path.join(self.args.save_folder, 'epoch_losses.npy'), epoch_losses)
            if self.best_model is not None:
                torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))

    def Test(self, testloader):
        self.eval()
        true_label, pred_label = [], []
        for iter, (x1, _, y1, _) in enumerate(testloader):
            x1 = x1.to(device)
            u1, _ = self.forward(x1)
            true_label.append(y1)
            pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label

    def Valid(self, validloader):
        self.eval()
        true_label, pred_label = [], []
        for iter, (x1, _, y1, _) in enumerate(validloader):
            x1 = x1.to(device)
            u1, _ = self.forward(x1)
            true_label.append(y1)
            pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()
