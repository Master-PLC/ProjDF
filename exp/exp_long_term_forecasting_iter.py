import os
import time
import torch
import warnings
import yaml

from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch.nn as nn

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import MODEL_REQUIRES_CYCLE
from utils.fft_ot import cal_wasserstein
from utils.metrics_torch import metric_torch
from utils.ot_dist import *
from utils.tools import EarlyStopping, visual, Scheduler

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Iter(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len

    def _build_model(self, args=None):
        args = deepcopy(self.args)
        args.pred_len = 1
        model = self.model_dict[args.model].Model(args).float()

        pretrain_model_path = args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f'Loading pretrained model from {pretrain_model_path}')
            state_dict = torch.load(pretrain_model_path)
            model.load_state_dict(state_dict, strict=False)

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        return model

    def _get_data(self, flag, shuffle=None):
        if flag == 'train':
            args = deepcopy(self.args)
            args.pred_len = 1
        else:
            args = self.args
        data_set, data_loader = data_provider(args, flag, shuffle=shuffle)
        return data_set, data_loader

    def infer_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        # 数据准备
        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
            batch_x_mark, batch_y_mark = None, None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        if batch_x_mark is not None:
            full_marks = torch.cat([batch_x_mark, batch_y_mark[:, self.label_len:, :]], dim=1)
        else:
            full_marks = None

        curr_x = batch_x
        seq_len = self.args.seq_len
        label_len = self.args.label_len

        preds = []
        for i in range(self.pred_len):
            dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]], device=self.device)
            dec_inp = torch.cat([curr_x[:, -label_len:, :], dec_inp], dim=1)

            if full_marks is not None:
                cur_x_mark = full_marks[:, i:i + seq_len, :]
                enc_end = i + seq_len
                cur_y_mark = full_marks[:, enc_end - label_len:enc_end + 1, :]
            else:
                cur_x_mark, cur_y_mark = None, None

            model_args = [curr_x, cur_x_mark, dec_inp, cur_y_mark]
            if self.args.model in MODEL_REQUIRES_CYCLE:
                model_args.append(batch_cycle)

            if self.args.output_attention:
                out, attn = self.model(*model_args)
            else:
                out, attn = self.model(*model_args), None
            
            step_pred = out[:, -1:, :]
            preds.append(step_pred)
            curr_x = torch.cat([curr_x, step_pred], dim=1)
            if curr_x.shape[1] > seq_len:
                curr_x = curr_x[:, -seq_len:, :]

        outputs = torch.cat(preds, dim=1)

        # 截取最终结果
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]

        return outputs, batch_y, attn

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                outputs, batch_y, _ = self.infer_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        # total_loss = np.average(total_loss)
        total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.report_to != 'None':
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
            train_loss = []

            lr_cur = scheduler.get_lr()
            lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur
            if self.writer is not None:
                self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                batch_y = batch_y[:, -1:, :]

                loss = 0
                if self.args.rec_lambda:
                    loss_rec = criterion(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)
                if self.step % self.log_step == 0 and self.writer is not None:
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss_rec, self.step)

                if self.args.l1_weight and attn:
                    loss += self.args.l1_weight * attn[0]

                if self.args.auxi_lambda:
                    if self.args.joint_forecast:  # joint distribution forecasting
                        outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1)  # [B, S+P, D]
                        batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1)  # [B, S+P, D]

                    if self.args.auxi_mode == "rfft":
                        if self.args.auxi_type == 'complex':
                            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)  # shape: [B, P//2+1, D]
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "fft_ot":
                        loss_auxi = cal_wasserstein(
                            outputs, batch_y, self.args.distance, ot_type=self.args.ot_type, normalize=self.args.normalize, 
                            mask_factor=self.args.mask_factor, reg_sk=self.args.reg_sk, stopThr=self.args.stopThr, numItermax=self.args.numItermax, 
                            var_weight=self.args.var_weight, mean_weight=self.args.mean_weight
                        )

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

                    loss += self.args.auxi_lambda * loss_auxi
                else:
                    loss_auxi = torch.tensor(1e4)
                if self.step % self.log_step == 0 and self.writer is not None:
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi, self.step)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Loss is NaN or Inf, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                train_loss.append(loss.item())
                if self.writer is not None:
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_iter', loss.item(), self.step)

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
                    model_state_last_effective = deepcopy(self.model.state_dict())  # save the last effective model state dict

                loss.backward()
                model_optim.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            if model_state_last_effective is not None and has_nan_in_epoch:
                self.model.load_state_dict(model_state_last_effective)

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            if self.writer is not None:
                self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
                self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))

        inputs, preds, trues = [], [], []
        if self.output_vis:
            folder_path = os.path.join(self.args.test_results, setting)
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                outputs, batch_y, _ = self.infer_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

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
        if self.report_to != 'None' and self.writer is None:
            self.writer = self._create_writer(res_path)

        metrics = OrderedDict()
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        metrics['mae'] = mae; metrics['mse'] = mse; metrics['rmse'] = rmse; metrics['mape'] = mape; metrics['mspe'] = mspe; metrics['mre'] = mre

        extra_metrics = OrderedDict()
        if self.args.extra_metrics != []:
            if any([x in self.args.extra_metrics for x in ['ot_dist', 'ot_dist_exact', 'wst1d']]):
                _preds = torch.cat([inputs, preds], dim=1)
                _trues = torch.cat([inputs, trues], dim=1)

            if 'ot_dist' in self.args.extra_metrics:
                ot_dist = cal_wasserstein(
                    _preds, _trues, "wasserstein_empirical_per_dim", ot_type="upper_bound", normalize=1, mask_factor=0.0, 
                    reg_sk=0.005, stopThr=1e-4, numItermax=10000, var_weight=0.00002, mean_weight=1.0, reweight=True
                )
                extra_metrics['ot_dist'] = ot_dist.item()
            if 'ot_dist_exact' in self.args.extra_metrics:
                ot_dist_exact = cal_wasserstein(
                    _preds, _trues, "emd_per_dim", normalize=1, norm_factor='T', mask_factor=0.2, numItermax=10000
                )
                extra_metrics['ot_dist_exact'] = ot_dist_exact.item()
            if 'wst1d' in self.args.extra_metrics:
                wst1d = cal_wasserstein(_preds, _trues, "wasserstein_1d_per_dim")
                extra_metrics['wst1d'] = wst1d.item()

        full_metrics = OrderedDict(**metrics, **extra_metrics)
        line = f'{self.args.data_id} @ {self.pred_len}\t| mse:{mse} mae:{mae}'
        if self.args.extra_metrics != []:
            extra_line = ', '.join([f'{k}:{v}' for k, v in extra_metrics.items()])
            line = f'{line}\t| {extra_line}'
        print(line)

        if self.writer is not None:
            for k, v in full_metrics.items():
                self.writer.add_scalar(f'{self.pred_len}/test/{k}', v, self.epoch)
            self.writer.close()

        if self.output_log:
            log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
            payload = f"{setting}\n\n{line}\n\n"
            with open(log_path, mode="a", encoding="utf-8") as f:
                f.write(payload)

        # np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre, ot_dist, ot_dist_exact, wst1d]))
        yaml.safe_dump(dict(full_metrics), open(os.path.join(res_path, 'metrics.yaml'), 'w'), default_flow_style=False, sort_keys=False)

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())
            if self.args.auxi_mode == 'basis' and self.args.auxi_type == 'pca':
                train_data, _ = self._get_data(flag='train')
                pca_components = train_data.pca_components
                np.save(os.path.join(res_path, 'pca_components.npy'), pca_components)

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            yaml.dump(vars(self.args), open(os.path.join(res_path, 'config.yaml'), 'w'), default_flow_style=False)

        return
