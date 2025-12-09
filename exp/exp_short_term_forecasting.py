import os
import pandas
import pickle
import time
import torch
import warnings
import yaml

import numpy as np
from torch import optim
import torch.nn as nn

from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from models import MODEL_REQUIRES_CYCLE
from utils.fft_ot import cal_wasserstein
from utils.fourier_koopman import fourier_loss
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
from utils.ot_dist import *
from utils.polynomial import chebyshev_torch, hermite_torch, laguerre_torch, leg_torch, pca_torch, Basis_Cache
from utils.soft_dtw_cuda import SoftDTW
from utils.tools import EarlyStopping, Scheduler, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        self.seasonal_patterns = args.seasonal_patterns
        super().__init__(args)

        self.pred_len = self.args.pred_len
        self.label_len = self.args.label_len
        assert self.args.output_attention == False

    def _build_model(self):
        if 'm4' in self.args.data:
            self.args.pred_len = M4Meta.horizons_map[self.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.seasonal_patterns]
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def initialize_cache(self, train_data):
        cache = None
        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'random':
                cache = Random_Cache(
                    rank_ratio=self.args.rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.pred_len, 
                    enc_in=self.args.enc_in, device=self.device
                )
            elif self.args.auxi_type == 'fa':
                cache = Basis_Cache(train_data.fa_components, train_data.initializer, mean=train_data.fa_mean, device=self.device)
            elif self.args.auxi_type == 'pca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, weights=train_data.weights, device=self.device)
            elif self.args.auxi_type == 'robustpca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, mean=train_data.rpca_mean, device=self.device)
            elif self.args.auxi_type == 'svd':
                cache = Basis_Cache(train_data.svd_components, train_data.initializer, device=self.device)
            elif self.args.auxi_type == 'ica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, mean=train_data.ica_mean, whitening=train_data.whitening, device=self.device)
            elif self.args.auxi_type == 'robustica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, device=self.device)
        return cache

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        model_args = [batch_x, None, dec_inp, None]
        if self.args.model in MODEL_REQUIRES_CYCLE:
            model_args.append(batch_cycle)
        outputs, attn = self.model(*model_args), None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return outputs, batch_y, attn

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        cache = self.initialize_cache(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.report_to == 'tensorboard':
            self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        if self.args.auxi_mode == 'fourier_koopman':
            freqs = nn.Parameter(torch.tensor(train_data.freqs, device=self.device, dtype=torch.float32))
            model_optim.add_param_group({'params': freqs, 'lr': self.args.learning_rate})
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion()
        mse = nn.MSELoss()
        if self.args.auxi_mode == 'soft_dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "SoftDTW only supports GPU"
            sdtw = SoftDTW(use_cuda=True, gamma=0.1)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []

            lr_cur = scheduler.get_lr()
            lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur
            if self.writer is not None:
                self.writer.add_scalar(f'{self.seasonal_patterns}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_mark = batch_y_mark[:, -self.pred_len:, f_dim:].to(self.device)

                loss = 0  # + loss_sharpness * 1e-5
                if self.args.rec_lambda:
                    if self.args.loss.lower() == 'smape':
                        loss_rec = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                    else:
                        loss_rec = criterion(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)

                if self.step % self.log_step == 0 and self.writer is not None:
                    self.writer.add_scalar(f'{self.seasonal_patterns}/train/loss_rec', loss_rec, self.step)

                if self.args.auxi_lambda:
                    if self.args.auxi_mode == "fft":
                        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)  # shape: [B, P, D]

                    elif self.args.auxi_mode == "rfft":
                        if self.args.auxi_type == 'complex':
                            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)  # shape: [B, P//2+1, D]
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "basis":
                        kwargs = {'degree': self.args.leg_degree, 'device': self.device}
                        if self.args.auxi_type == "random":
                            kwargs = {'pca_dim': self.args.pca_dim, 'random_cache': cache, 'device': self.device}
                            loss_auxi = random_torch(outputs, **kwargs) - random_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "pca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': cache, 'use_weights': self.args.use_weights, 
                                'reinit': self.args.reinit, 'device': self.device
                            }
                            loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "fourier_koopman":
                        loss_auxi = fourier_loss(outputs, batch_y, freqs, device=self.device)

                    elif self.args.auxi_mode == "soft_dtw":
                        mu = batch_y.mean(dim=[1, 2], keepdim=True)
                        std = batch_y.std(dim=[1, 2], keepdim=True) + 1e-6  # 加 1e-6 防止除以 0

                        # 将 true 和 pred 都归一化到同一尺度
                        batch_y = (batch_y - mu) / std
                        outputs = (outputs - mu) / std  # 注意：通常用 true 的统计量来归一化 pred
                        loss_auxi = sdtw(outputs, batch_y)
                    else:
                        raise NotImplementedError

                    if self.args.auxi_loss == "MAE":
                        # MAE, 最小化element-wise error的模长
                        loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    elif self.args.auxi_loss == "MSE":
                        # MSE, 最小化element-wise error的模长
                        loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                    elif self.args.auxi_loss == "None":
                        pass
                    else:
                        raise NotImplementedError

                    if torch.isnan(loss_auxi):
                        loss_auxi = torch.tensor(0, device=self.device)

                    loss += self.args.auxi_lambda * loss_auxi
                else:
                    loss_auxi = torch.tensor(1e4)

                if self.step % self.log_step == 0 and self.writer is not None:
                    self.writer.add_scalar(f'{self.seasonal_patterns}/train/loss_auxi', loss_auxi, self.step)

                train_loss.append(loss.item())
                if self.writer is not None:
                    self.writer.add_scalar(f'{self.seasonal_patterns}/train/loss_iter', loss.item(), self.step)

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {}, epoch: {} | loss_rec: {:.7f}, loss_auxi: {:.7f}, loss: {:.7f}".format(
                            i + 1, self.epoch, loss_rec.item(), loss_auxi.item(), loss.item()
                        )
                    )
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            if self.writer is not None:
                self.writer.add_scalar(f'{self.seasonal_patterns}/train/loss', train_loss, self.epoch)
                self.writer.add_scalar(f'{self.seasonal_patterns}/vali/loss', vali_loss, self.epoch)

            print(
                "Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    self.epoch, self.step, train_loss, vali_loss
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

    def vali(self, train_loader, vali_loader, criterion):
        x, _, cycle = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  # shape [B, S, 1]

        self.model.eval()
        eval_time = time.time()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.label_len:, :], dec_inp], dim=1).float()

            # encoder - decoder
            outputs = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_x = x[id_list[i]:id_list[i + 1]]
                _dec_inp = dec_inp[id_list[i]:id_list[i + 1]]
                batch_cycle = cycle[id_list[i]:id_list[i + 1]]
                model_args = [batch_x, None, _dec_inp, None]
                if self.args.model in MODEL_REQUIRES_CYCLE:
                    model_args.append(batch_cycle)
                _outputs, attn = self.model(*model_args), None
                outputs[id_list[i]:id_list[i + 1], :, :] = _outputs.detach()

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.pred_len:, f_dim:]

            pred = outputs
            true = torch.from_numpy(np.array(y)).to(self.device)
            true = true.unsqueeze(-1)  # shape [B, T, 1]
            batch_y_mark = torch.ones(true.shape).to(self.device)

            if self.args.loss.lower() == 'smape':
                loss = criterion(x.detach(), self.args.frequency_map, pred, true, batch_y_mark)
            else:
                loss = criterion(pred, true)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        self.model.train()
        return loss.item()

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')

        x, _, cycle = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  # shape [B, S, 1]

        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))

        if self.output_vis:
            folder_path = os.path.join(self.args.test_results, setting)
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.label_len:, :], dec_inp], dim=1).float()

            # encoder - decoder
            outputs = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_x = x[id_list[i]:id_list[i + 1]]
                _dec_inp = dec_inp[id_list[i]:id_list[i + 1]]
                batch_cycle = cycle[id_list[i]:id_list[i + 1]]
                model_args = [batch_x, None, _dec_inp, None]
                if self.args.model in MODEL_REQUIRES_CYCLE:
                    model_args.append(batch_cycle)
                _outputs, attn = self.model(*model_args), None
                outputs[id_list[i]:id_list[i + 1], :, :] = _outputs

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.pred_len:, f_dim:]
            outputs = outputs.detach()

            preds = outputs
            trues = torch.from_numpy(np.array(y)).to(self.device)
            x = x.detach()

            if self.output_vis:
                for i in range(0, preds.shape[0], preds.shape[0] // 10):
                    gt = np.concatenate((x[i, :, 0].cpu().numpy(), trues[i].cpu().numpy()), axis=0)
                    pd = np.concatenate((x[i, :, 0].cpu().numpy(), preds[i, :, 0].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        summary_path = os.path.join(self.args.results, "m4_results")
        os.makedirs(summary_path, exist_ok=True)

        forecasts_df = pandas.DataFrame(preds[:, :, 0].cpu().numpy(), columns=[f'V{i+1}' for i in range(self.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(os.path.join(summary_path, self.seasonal_patterns + '_forecast.csv'))

        filenames = os.listdir(summary_path)
        if all([f"{x}_forecast.csv" in filenames for x in M4Meta.seasonal_patterns]):
            m4_summary = M4Summary(summary_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            metrics = {
                "smape": smape_results,
                "mape": mape,
                "mase": mase,
                "owa": owa_results
            }
            line = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            print(line)

            pickle.dump(metrics, open(os.path.join(summary_path, "metrics.pkl"), 'wb'))

            if self.output_log:
                log_path = "result_short_term_forecast.txt" if not self.args.log_path else self.args.log_path
                payload = f"{setting}\n\n{line}\n\n"
                with open(log_path, mode="a", encoding="utf-8") as f:
                    f.write(payload)

        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')

        if self.writer is not None:
            self.writer.close()

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            yaml.dump(vars(self.args), open(os.path.join(res_path, 'config.yaml'), 'w'), default_flow_style=False)

        return
