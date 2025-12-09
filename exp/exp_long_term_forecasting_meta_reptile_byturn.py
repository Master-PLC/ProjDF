import os
import time
import torch
import warnings
import yaml

from collections import OrderedDict
from itertools import cycle
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from utils.metrics_torch import metric_torch
from utils.tools import EarlyStopping, Scheduler, disable_grad, enable_grad, log_heatmap, split_dataset_with_overlap, visual

warnings.filterwarnings('ignore')


class CovarianceMatrix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.L_param = nn.Parameter(torch.eye(args.pred_len))
        self.eps = 1e-6
        self.auxi_loss = args.auxi_loss

    def _get_L(self):
        # 取下三角并在对角线加 eps，确保正定
        L = torch.tril(self.L_param)
        diag = torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps)
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + diag
        return L

    def forward(self):
        L = self._get_L()
        return L @ L.transpose(-1, -2)               # Σ = L Lᵀ

    def get_inverse(self):
        L = self._get_L()
        A = L @ L.transpose(-1, -2)
        return torch.linalg.inv(A)

    def get_loss(self, pred, target):
        L = self._get_L()  # [P, P] 下三角矩阵

        E = pred - target  # [B, P, D]
        E_flat = E.permute(0, 2, 1).reshape(-1, self.pred_len)  # [B*D, P]

        # 解线性方程组 Lx = E_flat，得到 x = L^{-1}E_flat
        # 使用三角求解器（L是下三角矩阵）
        x = torch.linalg.solve_triangular(
            L, 
            E_flat.T,  # 转置为 [P, B*D]
            upper=False, 
            unitriangular=False
        ).T  # 转置回 [B*D, P]

        # 计算二次型: x^T x
        if self.auxi_loss == 'MSE':
            loss = torch.mean(x ** 2)
        elif self.auxi_loss == 'MAE':
            loss = torch.mean(x.abs())
        else:
            raise AttributeError(f"No defined loss type for {self.auxi_loss}.")

        if self.args.reg_lambda > 0:
            # force A to be close to identity matrix
            # I = torch.eye(self.pred_len, device=self.device)
            # reg_loss = torch.norm(fast_A() - I, p='fro') ** 2
            # loss += self.args.reg_lambda * reg_loss

            Sigma = self.forward()  # [P, P]
            off_diag = Sigma - torch.diag_embed(torch.diagonal(Sigma))
            reg_loss = torch.norm(off_diag, p='fro') ** 2
            loss += self.args.reg_lambda * reg_loss

        return loss


def get_projection(A):
    with torch.no_grad():
        Am = A()
    return Am.detach().cpu().numpy()  # 返回 numpy 数组


class Exp_Long_Term_Forecast_META_Reptile_Byturn(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.n_inner = args.meta_inner_steps
        self.lr = args.learning_rate
        self.inner_lr = args.inner_lr
        self.meta_lr = args.meta_lr

        self.A = CovarianceMatrix(self.args).to(self.device)
        disable_grad(self.A)  # A is not trainable in the beginning

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_cov_loss = []
        self.model.eval()
        self.A.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                cov_loss = self.A.get_loss(pred, true)

                total_loss.append(loss)
                total_cov_loss.append(cov_loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
        total_cov_loss = torch.mean(torch.stack(total_cov_loss)).item()

        self.model.train()
        self.A.train()
        return total_loss, total_cov_loss

    def inner_loop(self, task_id, task_loader):
        losses = []

        fast_A = CovarianceMatrix(self.args).to(self.device)
        fast_A.load_state_dict(self.A.state_dict())  # copy A to fast_A
        fast_A.train()
        optimizer = self._select_optimizer(fast_A, lr=self.inner_lr, optim_type=self.args.meta_optim_type)

        for k in range(self.n_inner):
            bx, by, bx_mark, by_mark, by_cycle = next(task_loader)
            with torch.no_grad():
                outputs, batch_y, _ = self.forward_step(bx, by, bx_mark, by_mark, by_cycle)

            optimizer.zero_grad()
            loss = fast_A.get_loss(outputs, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(fast_A.parameters(), max_norm=self.args.max_norm)
            optimizer.step()

            losses.append(loss.item())

        fast_state = fast_A.state_dict()
        return np.mean(losses), fast_state

    def meta_update(self, task_states):
        for i, (name, p) in enumerate(self.A.named_parameters()):
            meta_grad = torch.zeros_like(p)
            for task_state in task_states:
                meta_grad += (task_state[name] - p).detach()
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.data.add_(meta_grad / len(task_states))

    def initialize_meta_tasks(self, train_data):
        self.meta_learning = False

        task_data_list = split_dataset_with_overlap(train_data, self.args.num_tasks, self.args.overlap_ratio)
        task_loader_list = [DataLoader(task_data, batch_size=self.args.auxi_batch_size, shuffle=True) for task_data in task_data_list]
        task_loader_list = [cycle(task_loader) for task_loader in task_loader_list]
        return task_loader_list

    def check_meta_learning(self):
        if self.args.fixed_step and self.step > self.args.fixed_step and not self.meta_learning:
            enable_grad(self.A)  # start training A
            print(f"\n>>>>>>>Meta learning enabled at step {self.step}, epoch {self.epoch}\n")
            self.meta_learning = True

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        task_loader_list = self.initialize_meta_tasks(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.report_to != 'None':
            self.writer = self._create_writer(res_path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer(self.model, self.lr)
        scheduler = Scheduler(model_optim, self.args, train_steps)

        A_optim = self._select_optimizer(self.A, self.meta_lr, optim_type=self.args.meta_optim_type)
        A_scheduler = Scheduler(A_optim, self.args, train_steps)

        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss, train_loss_mse, meta_loss = [], [], []

            lr_cur = scheduler.get_lr()
            meta_lr_cur = A_scheduler.get_lr()
            if self.writer is not None:
                self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)
                self.writer.add_scalar(f'{self.pred_len}/train/meta_lr', meta_lr_cur, self.epoch)

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.model.train()
                self.A.eval()

                self.step += 1
                iter_count += 1

                model_optim.zero_grad()
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                loss = self.A.get_loss(outputs, batch_y)
                loss.backward()
                model_optim.step()

                with torch.no_grad():
                    loss_mse = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                train_loss_mse.append(loss_mse.item())
                if self.writer is not None:
                    self.writer.add_scalar(f'{self.pred_len}/train_iter/loss_cov', loss.item(), self.step)
                    self.writer.add_scalar(f'{self.pred_len}/train_iter/loss_mse', loss_mse.item(), self.step)

                self.check_meta_learning()
                if self.meta_learning:
                    self.model.eval()
                    self.A.train()

                    A_optim.zero_grad()
                    task_losses, task_states = [], []
                    for k, task_loader in enumerate(task_loader_list):
                        task_loss, task_state = self.inner_loop(k, task_loader)
                        task_losses.append(task_loss)
                        task_states.append(task_state)
                    self.meta_update(task_states)
                    A_optim.step()

                    meta_l = np.mean(task_losses)
                else:
                    meta_l = 1e4

                meta_loss.append(meta_l)
                if self.writer is not None:
                    self.writer.add_scalar(f'{self.pred_len}/train_iter/loss_meta', meta_l, self.step)

                if (i + 1) % 100 == 0:
                    print("\titers: {}, epoch: {} | loss: {:.7f}, meta loss: {:.7f}".format(i + 1, self.epoch, loss.item(), meta_l))
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))
                    if self.meta_learning:
                        A_scheduler.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_mse = np.average(train_loss_mse)
            meta_loss = np.average(meta_loss)
            valid_loss_mse, valid_loss_cov = self.vali(vali_data, vali_loader, criterion)

            if self.writer is not None:
                self.writer.add_scalar(f'{self.pred_len}/train/loss_cov', train_loss, self.epoch)
                self.writer.add_scalar(f'{self.pred_len}/train/loss_mse', train_loss_mse, self.epoch)
                self.writer.add_scalar(f'{self.pred_len}/train/loss_meta', meta_loss, self.epoch)
                self.writer.add_scalar(f'{self.pred_len}/vali/loss_cov', valid_loss_cov, self.epoch)
                self.writer.add_scalar(f'{self.pred_len}/vali/loss_mse', valid_loss_mse, self.epoch)
                log_heatmap(self.writer, get_projection(self.A), f'{self.pred_len}/cov_mat', self.epoch)

            print(
                "Epoch: {}, Steps: {} | Train Loss Cov: {:.7f}, MSE: {:.7f}, Meta: {:.7f} | Vali Loss Cov: {:.7f}, MSE: {:.7f}".format(
                    self.epoch, self.step, train_loss, train_loss_mse, meta_loss, valid_loss_cov, valid_loss_mse
                )
            )
            other_to_save = {'A': self.A}
            improved = early_stopping(valid_loss_mse, self.model, path, **other_to_save)
            self.args.learned_from_method = True if improved and self.meta_learning else False

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(valid_loss_mse, self.epoch)
                if self.meta_learning:
                    A_scheduler.step(valid_loss_mse, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        best_A_path = os.path.join(path, 'A.pth')
        self.A = torch.load(best_A_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
            self.A = torch.load(os.path.join(ckpt_dir, 'A.pth'))

        inputs, preds, trues = [], [], []
        if self.output_vis:
            folder_path = os.path.join(self.args.test_results, setting)
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        self.A.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

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
            if 'cov' in self.args.extra_metrics:
                with torch.no_grad():
                    self.A.to(preds.device)
                    cov_loss = self.A.get_loss(preds, trues)
                extra_metrics['cov'] = cov_loss.item()

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

        # np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))
        yaml.safe_dump(dict(full_metrics), open(os.path.join(res_path, 'metrics.yaml'), 'w'), default_flow_style=False, sort_keys=False)

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            yaml.dump(vars(self.args), open(os.path.join(res_path, 'config.yaml'), 'w'), default_flow_style=False)

        return
