import os
import time
import warnings
from copy import deepcopy
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.profiler as profiler
import yaml
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from torch import optim
from torch.utils.data import DataLoader
from utils.fft_ot import cal_wasserstein
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.ot_dist import *
from utils.tools import EarlyStopping, Scheduler, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_OT(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        pretrain_model_path = self.args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f'Loading pretrained model from {pretrain_model_path}')
            state_dict = torch.load(pretrain_model_path)
            model.load_state_dict(state_dict, strict=False)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, lr=None):
        if self.args.optim_type == 'adam':
            optim_class = optim.Adam
        elif self.args.optim_type == 'adamw':
            optim_class = optim.AdamW
        model_optim = optim_class(self.model.parameters(), lr=self.args.learning_rate if lr is None else lr)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                _, outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        # total_loss = np.average(total_loss)
        total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
        self.model.train()
        return total_loss

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.output_attention:
            outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            # outputs shape: [B, P, D]
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            attn = None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return batch_x, outputs, batch_y, attn

    def train(self, setting, prof=None):
        train_data, train_loader = self._get_data(flag='train')
        auxi_loader = DataLoader(
            train_data, batch_size=self.args.auxi_batch_size, shuffle=True, 
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
        auxi_optim = self._select_optimizer(lr=self.args.inner_lr)
        scheduler = Scheduler(model_optim, self.args, train_steps)
        auxi_scheduler = Scheduler(auxi_optim, self.args, train_steps)
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            has_nan_in_epoch = False

            first_train_loss = []
            second_train_loss = []

            lr_cur = scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1

                model_optim.zero_grad()
                _, outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(outputs, batch_y)
                if self.step % self.log_step == 0:
                    self.writer.add_scalar(f'{self.pred_len}/first_train/loss_iter', loss, self.step)
                loss.backward()
                model_optim.step()
                first_train_loss.append(loss.item())

                auxi_batch_x, auxi_batch_y, auxi_batch_x_mark, auxi_batch_y_mark = next(auxi_train_loader)
                auxi_optim.zero_grad()
                auxi_batch_x, outputs, batch_y, _ = self.forward_step(auxi_batch_x, auxi_batch_y, auxi_batch_x_mark, auxi_batch_y_mark)

                if self.args.joint_forecast:  # joint distribution forecasting
                    outputs = torch.concat((auxi_batch_x, outputs), dim=1)  # [B, S+P, D]
                    batch_y = torch.concat((auxi_batch_x, batch_y), dim=1)  # [B, S+P, D]

                loss_auxi = cal_wasserstein(
                    outputs, batch_y, self.args.distance, ot_type=self.args.ot_type, normalize=self.args.normalize, 
                    mask_factor=self.args.mask_factor, reg_sk=self.args.reg_sk, stopThr=self.args.stopThr, numItermax=self.args.numItermax
                )

                if self.args.auxi_loss == "MAE":
                    # MAE, 最小化element-wise error的模长
                    loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                elif self.args.auxi_loss == "MSE":
                    # MSE, 最小化element-wise error的模长
                    loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                elif self.args.auxi_loss in ["None", "none"]:
                    pass
                else:
                    raise NotImplementedError

                if self.step % self.log_step == 0:
                    self.writer.add_scalar(f'{self.pred_len}/second_train/loss_iter', loss_auxi, self.step)

                if torch.isnan(loss_auxi) or torch.isinf(loss_auxi):
                    print(f"Loss is NaN or Inf in second train, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                loss_auxi.backward()
                auxi_optim.step()
                second_train_loss.append(loss_auxi.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {}, epoch: {} | 1st loss: {:.7f}, 2nd loss: {:.7f}".format(i + 1, self.epoch, loss.item(), loss_auxi.item()))
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()
                    model_state_last_effective = deepcopy(self.model.state_dict())  # save the last effective model state dict

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))
                    auxi_scheduler.step(verbose=(i + 1 == train_steps))

            if model_state_last_effective is not None and has_nan_in_epoch:
                self.model.load_state_dict(model_state_last_effective)

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            first_train_loss = np.average(first_train_loss)
            second_train_loss = np.average(second_train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/first_train/loss', first_train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/second_train/loss', second_train_loss, self.epoch)
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
                auxi_scheduler.step(vali_loss, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, prof=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                _, outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    batch_x = batch_x.cpu().numpy()
                    in_shape = batch_x.shape
                    batch_x = test_data.inverse_transform(batch_x.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x).float().to(self.device)

                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()
                    out_shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)

                inputs.append(batch_x.cpu())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if prof is not None:
                    prof.step()

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        # m = metric_collector.compute()
        # mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))

        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())

        print('save configs')
        args_dict = vars(self.args)
        with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
            yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return
