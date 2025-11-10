import torch
import torch.nn as nn
import numpy as np
from utils.util import AverageMeter, get_logger, eval_metrix
import os
from Model.Model_MLPtoKANs import KAN, Solution_u

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        if args.save_folder and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()

        self.model = Solution_u(u_hidden_dim=args.u_hidden_dim, u_layers_num=args.u_layers_num).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.warmup_lr)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.loss_func = nn.MSELoss()
        self.best_model = None

    def _save_args(self):
        if self.args.log_dir:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.info(f"\t{k}:{v}")

    def train_one_epoch(self, epoch, dataloader):
        self.model.train()
        loss_meter = AverageMeter()
        for idx, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, y1 = x1.to(device), y1.to(device)
            pred = self.model(x1)
            loss = self.loss_func(pred, y1)
            l2_reg = torch.tensor(0.).to(device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += self.args.l2_lambda * l2_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())
        return loss_meter.avg

    def save_model_architecture(self, metrics):
        architecture_path = os.path.join(self.args.save_folder, 'model_architecture.txt')
        with open(architecture_path, 'w') as f:
            f.write("Model: KANN + MLP\n\n")
            f.write("KAN Encoder:\n")
            f.write(f"  Input dimension: 17\n")
            f.write(f"  Output dimension: 32\n")
            f.write(f"  Number of hidden layers: {self.args.u_layers_num}\n")
            f.write(f"  Number of neurons in each hidden layer: {self.args.u_hidden_dim}\n\n")
            f.write("MLP Predictor:\n")
            f.write(f"  Input dimension: 32\n")
            f.write(f"  Hidden dimension: 32\n")
            f.write(f"  Output dimension: 1\n\n")
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
            train_loss = self.train_one_epoch(e, trainloader)
            self.scheduler.step()
            epoch_losses.append(train_loss)

            info = f'[Train] epoch:{e}, lr:{self.optimizer.param_groups[0]["lr"]:.6f}, total loss:{train_loss:.6f}'
            self.logger.info(info)

            if e > 1 and train_loss > epoch_losses[-2]:
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
                self.best_model = self.model.state_dict()
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
        self.model.eval()
        true_label, pred_label = [], []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.model(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        return true_label, pred_label

    def Valid(self, validloader):
        self.model.eval()
        true_label, pred_label = [], []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.model(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()
