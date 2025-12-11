import os
import time
import torch
import warnings


from copy import deepcopy
from itertools import cycle
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from utils.fft_ot import cal_wasserstein
from utils.ot_dist import *
from utils.tools import EarlyStopping, Scheduler

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_OT(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        auxi_loader = DataLoader(
            train_data, batch_size=self.args.auxi_batch_size - self.args.batch_size, shuffle=True, 
            num_workers=self.args.num_workers, drop_last=True
        )
        auxi_train_loader = cycle(auxi_loader)  # cycle the auxiliary loader
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        model_state_last_effective = None
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            has_nan_in_epoch = False

            first_train_loss = []
            second_train_loss = []
            train_loss = []

            lr_cur = scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1

                model_optim.zero_grad()
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                loss = criterion(outputs, batch_y)
                self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss.item(), self.step)
                first_train_loss.append(loss.item())

                auxi_batch_x, auxi_batch_y, auxi_batch_x_mark, auxi_batch_y_mark = next(auxi_train_loader)

                auxi_batch_x = torch.concat([batch_x, auxi_batch_x.to(batch_x.device)], dim=0)
                auxi_batch_y = torch.concat([batch_y, auxi_batch_y.to(batch_y.device)], dim=0)
                auxi_batch_x_mark = torch.concat([batch_x_mark, auxi_batch_x_mark.to(batch_x_mark.device)], dim=0)
                auxi_batch_y_mark = torch.concat([batch_y_mark, auxi_batch_y_mark.to(batch_y_mark.device)], dim=0)

                outputs, batch_y, _ = self.forward_step(auxi_batch_x, auxi_batch_y, auxi_batch_x_mark, auxi_batch_y_mark)

                if self.args.joint_forecast:  # joint distribution forecasting
                    outputs = torch.concat((auxi_batch_x.to(outputs.device), outputs), dim=1)  # [B, S+P, D]
                    batch_y = torch.concat((auxi_batch_x.to(batch_y.device), batch_y), dim=1)  # [B, S+P, D]

                loss_auxi = cal_wasserstein(
                    outputs, batch_y, self.args.distance, ot_type=self.args.ot_type, normalize=self.args.normalize, 
                    mask_factor=self.args.mask_factor, reg_sk=self.args.reg_sk, stopThr=self.args.stopThr, numItermax=self.args.numItermax, var_weight=self.args.var_weight
                )
                self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi.item(), self.step)
                second_train_loss.append(loss_auxi.item())

                if torch.isnan(loss_auxi) or torch.isinf(loss_auxi):
                    print(f"Loss is NaN or Inf in second train, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                loss += self.args.auxi_lambda * loss_auxi
                self.writer.add_scalar(f'{self.pred_len}/train/loss_iter', loss.item(), self.step)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {}, epoch: {} | loss_rec: {:.7f}, loss_auxi: {:.7f}, loss: {:.7f}".format(
                            i + 1, self.epoch, first_train_loss[-1], second_train_loss[-1], loss.item()
                        )
                    )
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()
                    model_state_last_effective = deepcopy(self.model.state_dict())  # save the last effective model state dict

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            if model_state_last_effective is not None and has_nan_in_epoch:
                self.model.load_state_dict(model_state_last_effective)

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            first_train_loss = np.average(first_train_loss)
            second_train_loss = np.average(second_train_loss)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', first_train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', second_train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)

            print(
                "Epoch: {}, Steps: {} | 1st Train Loss: {:.7f} 2nd Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    self.epoch, self.step, first_train_loss, second_train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(vali_loss, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
