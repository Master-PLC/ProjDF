import os
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from torch import optim
from utils.cca_loss import cca_loss
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.tools import EarlyStopping, Scheduler, adjust_learning_rate, disable_grad, enable_grad, log_heatmap, visual

warnings.filterwarnings('ignore')


def get_projection(proj, proj_init='identity'):
    if proj_init == 'linear':
        return proj.weight.data.cpu().numpy()
    elif proj_init == 'mlp':
        proj0 = proj[0].weight.data
        proj1 = proj[2].weight.data
        return (proj0 @ proj1).cpu().numpy()
    else:
        return proj.data.cpu().numpy()


class Exp_Long_Term_Forecast_CCA_Loss(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len

    def _build_model(self):
        args = deepcopy(self.args)
        self.proj_dim = int(args.enc_in * args.rank_ratio) if args.rank_ratio and args.rank_ratio <= 1 else int(abs(args.rank_ratio))
        args.enc_in = self.proj_dim
        args.dec_in = self.proj_dim
        args.c_out = self.proj_dim

        model = self.model_dict[args.model].Model(args).float()

        pretrain_model_path = args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f'Loading pretrained model from {pretrain_model_path}')
            state_dict = torch.load(pretrain_model_path)
            model.load_state_dict(state_dict, strict=False)

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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
                predictions, outputs, batch_x, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

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
        if self.projection_learning:
            if self.args.proj_init == 'cca' and self.args.pre_norm:
                batch_x = (batch_x - self.means[0]) / self.stds[0]  # [B, S, D]
            if self.args.proj_init in ['linear', 'mlp']:
                batch_x = self.x_proj(batch_x)
            else:
                batch_x = torch.matmul(batch_x, self.x_proj)  # [B, S, D] -> [B, S, rank]
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
        predictions = outputs = outputs[:, -self.pred_len:, f_dim:]
        if self.projection_learning:
            if self.args.proj_init in ['linear', 'mlp']:
                outputs = self.y_proj(outputs)  # [B, P, rank] -> [B, P, D]
            else:
                outputs = torch.matmul(outputs, self.y_proj)
            if self.args.proj_init == 'cca' and self.args.pre_norm:
                outputs = outputs * self.stds[1] + self.means[1]  # inverse transform outputs, mul std and add mean
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return predictions, outputs, batch_x, batch_y, attn

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')

        if self.args.proj_init == 'identity':
            x_proj = torch.eye(self.args.enc_in, self.proj_dim, dtype=torch.float32).to(self.device)
            y_proj = torch.eye(self.proj_dim, self.args.dec_in, dtype=torch.float32).to(self.device)
            if self.args.identity_direction == 'right':
                x_proj = x_proj.flip(1)
                y_proj = y_proj.flip(1)

        elif self.args.proj_init == 'linear':
            self.x_proj = nn.Linear(self.args.enc_in, self.proj_dim).to(self.device)
            self.y_proj = nn.Linear(self.proj_dim, self.args.dec_in).to(self.device)

        elif self.args.proj_init == 'mlp':
            self.x_proj = nn.Sequential(
                nn.Linear(self.args.enc_in, self.args.enc_in),
                nn.Sigmoid(),
                nn.Linear(self.args.enc_in, self.proj_dim)
            ).to(self.device)
            self.y_proj = nn.Sequential(
                nn.Linear(self.proj_dim, self.args.dec_in),
                nn.Sigmoid(),
                nn.Linear(self.proj_dim, self.args.dec_in)
            ).to(self.device)

        elif self.args.proj_init == 'random':
            x_proj = torch.randn((self.args.enc_in, self.proj_dim), dtype=torch.float32).to(self.device)
            y_proj = torch.randn((self.proj_dim, self.args.dec_in), dtype=torch.float32).to(self.device)

        elif self.args.proj_init == 'cca':
            assert '_CCA' in self.args.data, "CCA projection initialization requires '_CCA' in dataset name"
            x_proj, y_proj = train_data.Wx, train_data.Wy.T  # [D, rank] and [rank, D]
            x_proj = torch.as_tensor(x_proj).float().to(self.device)
            y_proj = torch.as_tensor(y_proj).float().to(self.device)
            self.means = train_data.means
            self.stds = train_data.stds
            self.means = [torch.as_tensor(m).float().to(self.device) for m in self.means]
            self.stds = [torch.as_tensor(s).float().to(self.device) for s in self.stds]

        if self.args.proj_init in ['linear', 'mlp']:
            disable_grad(self.x_proj)
            disable_grad(self.y_proj)
        else:
            self.x_proj = nn.Parameter(x_proj, requires_grad=False)
            self.y_proj = nn.Parameter(y_proj, requires_grad=False)

        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        if self.args.proj_init in ['linear', 'mlp']:
            proj_params = list(self.x_proj.parameters()) + list(self.y_proj.parameters())
        else:
            proj_params = [self.x_proj, self.y_proj]
        model_optim.add_param_group({'params': proj_params, 'lr': self.args.inner_lr})
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion()

        self.projection_learning = False
        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []

            if self.args.fixed_epoch and self.epoch > self.args.fixed_epoch and not self.projection_learning:
                if self.args.learn_x_proj:
                    enable_grad(self.x_proj)
                if self.args.learn_y_proj:
                    enable_grad(self.y_proj)
                print(f"\n>>>>>>>Projection learning enabled at epoch {self.epoch}, step {self.step}\n")
                self.projection_learning = True

            lr_cur = scheduler.get_lr()
            if isinstance(lr_cur, list):
                for lr_idx, lr in enumerate(lr_cur):
                    self.writer.add_scalar(f'{self.pred_len}/train/lr_{lr_idx}', lr, self.epoch)
            else:
                self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1

                if self.args.fixed_step and self.step > self.args.fixed_step and not self.projection_learning:
                    if self.args.learn_x_proj:
                        enable_grad(self.x_proj)
                    if self.args.learn_y_proj:
                        enable_grad(self.y_proj)
                    print(f"\n>>>>>>>Projection learning enabled at step {self.step}, epoch {self.epoch}\n")
                    self.projection_learning = True

                model_optim.zero_grad()

                predictions, outputs, batch_x, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = 0
                if self.args.rec_lambda:
                    loss_rec = criterion(outputs, batch_y)
                    if self.projection_learning:
                        loss += self.args.rec_lambda * loss_rec
                    else:
                        loss += loss_rec
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss_rec, self.step)

                if self.args.auxi_lambda:
                    if self.args.joint_forecast:  # joint distribution forecasting
                        outputs = torch.concat((batch_x, outputs), dim=1)  # [B, S+P, D]
                        batch_y = torch.concat((batch_x, batch_y), dim=1)  # [B, S+P, D]

                    if self.args.auxi_mode == "cca":
                        if self.projection_learning:
                            loss_auxi = cca_loss(
                                batch_x, predictions, align_type=int(self.args.auxi_type), rank_ratio=self.args.rank_ratio, 
                                device=self.device, r1=self.args.reg_cca, r2=self.args.reg_cca, eps=self.args.eps, loss_type=self.args.cca_type
                            )
                        else:
                            loss_auxi = 0

                    else:
                        raise NotImplementedError

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

                    loss += self.args.auxi_lambda * loss_auxi
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi, self.step)

                if self.args.reg_lambda and self.projection_learning:
                    if self.args.cca_type == 'cosine':
                        reg_loss_x = torch.norm(batch_x, p=2)
                        reg_loss_y = torch.norm(predictions, p=2)
                    else:
                        I_x = torch.eye(self.args.enc_in, self.proj_dim, device=self.device)
                        I_y = torch.eye(self.proj_dim, self.args.dec_in, device=self.device)
                        if self.args.identity_direction == 'right':
                            I_x = I_x.flip(1)
                            I_y = I_y.flip(1)
                        if self.args.proj_init == 'linear':
                            reg_loss_x = torch.norm(self.x_proj.weight - I_x, p='fro') ** 2
                            reg_loss_y = torch.norm(self.y_proj.weight - I_y, p='fro') ** 2
                        elif self.args.proj_init == 'mlp':
                            reg_loss_x = torch.norm(self.x_proj[2].weight - I_x, p='fro') ** 2
                            reg_loss_y = torch.norm(self.y_proj[2].weight - I_y, p='fro') ** 2
                        else:
                            reg_loss_x = torch.norm(self.x_proj - I_x, p='fro') ** 2
                            reg_loss_y = torch.norm(self.y_proj - I_y, p='fro') ** 2
                    reg_loss = 0
                    if self.args.learn_x_proj:
                        reg_loss += reg_loss_x
                    if self.args.learn_y_proj:
                        reg_loss += reg_loss_y

                    loss += self.args.reg_lambda * reg_loss
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_reg', reg_loss.item(), self.step)

                train_loss.append(loss.item())
                self.writer.add_scalar(f'{self.pred_len}/train/loss_iter', loss.item(), self.step)

                if (i + 1) % 100 == 0:
                    print("\titers: {}, epoch: {} | loss: {:.7f}".format(i + 1, self.epoch, loss.item()))
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # nn.utils.clip_grad_norm_([p for group in model_optim.param_groups for p in group['params']], self.args.grad_clip)
                model_optim.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)
            log_heatmap(self.writer, get_projection(self.x_proj, self.args.proj_init), f'{self.pred_len}/x_proj', self.epoch)
            log_heatmap(self.writer, get_projection(self.y_proj, self.args.proj_init), f'{self.pred_len}/y_proj', self.epoch)

            print(
                "Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    self.epoch, self.step, train_loss, vali_loss
                )
            )
            other_to_save = {'x_proj': self.x_proj, 'y_proj': self.y_proj}
            if self.args.proj_init == 'cca' and self.args.pre_norm:
                other_to_save['means'] = self.means
                other_to_save['stds'] = self.stds
            early_stopping(vali_loss, self.model, path, **other_to_save)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(vali_loss, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.x_proj = torch.load(os.path.join(path, 'x_proj.pth'))
        self.y_proj = torch.load(os.path.join(path, 'y_proj.pth'))
        if self.args.proj_init == 'cca' and self.args.pre_norm:
            self.means = torch.load(os.path.join(path, 'means.pth'))
            self.stds = torch.load(os.path.join(path, 'stds.pth'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
            self.projection_learning = True
            self.x_proj = torch.load(os.path.join(ckpt_dir, 'x_proj.pth'))
            self.y_proj = torch.load(os.path.join(ckpt_dir, 'y_proj.pth'))
            if self.args.proj_init == 'cca' and self.args.pre_norm:
                self.means = torch.load(os.path.join(ckpt_dir, 'means.pth'))
                self.stds = torch.load(os.path.join(ckpt_dir, 'stds.pth'))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                _, outputs, _, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

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
