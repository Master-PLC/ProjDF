import os
import time
import warnings
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from utils.metrics_torch import metric_torch
from utils.metrics import metric  # 如果没用可移除
from utils.tools import (
    EarlyStopping,
    Scheduler,
    clip_grads,
    disable_grad,
    enable_grad,
    log_heatmap,
    split_dataset,
    split_dataset_with_overlap,
    visual
)

warnings.filterwarnings('ignore')


class CovarianceMatrix(nn.Module):
    """
    可学习的协方差（通过下三角 L 构造 Σ = L L^T），
    用以定义一个可学习的时间维度相关性损失。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.L_param = nn.Parameter(torch.eye(args.pred_len))
        self.eps = 1e-6
        self.auxi_loss = getattr(args, 'auxi_loss', 'MSE')

    def _get_L(self, params=None):
        if params is None:
            L_param = self.L_param
        else:
            L_param = params['L_param']
        L = torch.tril(L_param)
        diag = torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps)
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + diag
        return L

    def forward(self, params=None):
        L = self._get_L(params)
        return L @ L.transpose(-1, -2)

    def get_inverse(self, params=None):
        L = self._get_L(params)
        A = L @ L.transpose(-1, -2)
        return torch.linalg.inv(A)

    def get_loss(self, pred, target, params=None):
        """
        pred / target: [B, pred_len, D]
        使用 L 来对 (pred - target) 做“白化”后再计算加权误差。
        """
        L = self._get_L(params)
        E = pred - target  # [B, P, D]
        # 重排为 [B*D, P]
        E_flat = E.permute(0, 2, 1).reshape(-1, self.pred_len)
        # 解 L x = E^T
        x = torch.linalg.solve_triangular(
            L,
            E_flat.T,
            upper=False,
            unitriangular=False
        ).T
        if self.auxi_loss == 'MSE':
            loss = torch.mean(x ** 2)
        elif self.auxi_loss == 'MAE':
            loss = torch.mean(x.abs())
        else:
            raise AttributeError(f"No defined loss type for {self.auxi_loss}.")

        reg_lambda = getattr(self.args, 'reg_lambda', 0.0)
        if reg_lambda > 0:
            Sigma = self.forward(params)
            off_diag = Sigma - torch.diag_embed(torch.diagonal(Sigma))
            reg_loss = torch.norm(off_diag, p='fro') ** 2
            loss += reg_lambda * reg_loss
        return loss


def get_projection(A):
    with torch.no_grad():
        Am = A()
    return Am.detach().cpu().numpy()


def get_param_dict(module):
    return dict(module.named_parameters())


def update_param_dict(param_dict, grads, lr):
    return {k: v - lr * grads[k] for k, v in param_dict.items()}


class Exp_Long_Term_Forecast_META_V1(Exp_Basic):
    """
    两阶段版本：
    Phase 1: Meta Learning 训练 A （协方差结构/元损失），冻结 base model
    Phase 2: 使用已训练好的 A (固定) 作为损失，对 base model 做普通训练
    """
    def __init__(self, args):
        super().__init__(args)
        # 任务相关超参
        self.pred_len = args.pred_len
        self.label_len = args.label_len

        # Meta 超参
        self.n_inner = getattr(args, 'meta_inner_steps', 5)
        self.inner_lr = getattr(args, 'inner_lr', 1e-3)
        self.meta_lr = getattr(args, 'meta_lr', 1e-3)
        self.first_order = getattr(args, 'first_order', False)
        self.num_tasks = getattr(args, 'num_tasks', 4)
        self.overlap_ratio = getattr(args, 'overlap_ratio', 0.2)
        self.auxi_batch_size = getattr(args, 'auxi_batch_size', 32)

        # Phase 1 epochs / early stop
        self.meta_epochs = getattr(args, 'meta_epochs', 30)
        self.meta_patience = getattr(args, 'meta_patience', 5)

        # Phase 2 (base training) 超参
        self.lr = getattr(args, 'learning_rate', 1e-3)
        self.train_epochs = getattr(args, 'train_epochs', 10)

        # 其余
        self.max_norm = getattr(args, 'max_norm', None)

        # 协方差模块 A
        self.A = CovarianceMatrix(self.args).to(self.device)

        # Writer 与计数器
        self.meta_global_step = 0
        self.train_global_step = 0
        self.writer = None

        # 标记
        self.meta_trained = False

    # =============== Meta Learning Phase (Train A) ===============

    def initialize_meta_tasks(self, train_data):
        """
        将训练数据切分成多个任务，并为每个任务建立 support/query loader
        """
        task_data_list = split_dataset_with_overlap(
            train_data,
            self.num_tasks,
            self.overlap_ratio
        )
        # 每个 task 再 7:3 划分 support/query
        task_data_list = [split_dataset(task_data, r=0.7) for task_data in task_data_list]

        support_data_list = [td[0] for td in task_data_list]
        support_loader_list = [
            DataLoader(support_data, batch_size=self.auxi_batch_size, shuffle=True)
            for support_data in support_data_list
        ]
        support_loader_list = [cycle(ld) for ld in support_loader_list]

        query_data_list = [td[1] for td in task_data_list]
        query_loader_list = [
            DataLoader(query_data, batch_size=self.auxi_batch_size, shuffle=True)
            for query_data in query_data_list
        ]
        query_loader_list = [cycle(ld) for ld in query_loader_list]

        return support_loader_list, query_loader_list

    def inner_loop(self, support_loader, query_loader):
        """
        针对一个任务的 MAML inner adaptation:
        - fast_params: A 的虚拟参数副本
        - 多步 support set 更新
        - 最终在 query set 上计算 meta 梯度
        返回: support 平均 loss, query loss (tensor)
        """
        fast_params = get_param_dict(self.A)
        support_losses = []

        for _ in range(self.n_inner):
            bx, by, bx_mark, by_mark, by_cycle = next(support_loader)
            # base model 冻结，用于生成预测
            with torch.no_grad():
                outputs, batch_y, _ = self.forward_step(
                    bx, by, bx_mark, by_mark, by_cycle
                )
            loss_s = self.A.get_loss(outputs, batch_y, params=fast_params)
            grads = torch.autograd.grad(
                loss_s,
                fast_params.values(),
                create_graph=not self.first_order
            )
            if self.max_norm is not None:
                grads = clip_grads(grads, self.max_norm)
            grads_dict = {k: g for k, g in zip(fast_params.keys(), grads)}
            fast_params = update_param_dict(fast_params, grads_dict, self.inner_lr)
            support_losses.append(loss_s.detach().item())

        # Query
        bxq, byq, bxq_mark, byq_mark, byq_cycle = next(query_loader)
        with torch.no_grad():
            outputs_q, batch_y_q, _ = self.forward_step(
                bxq, byq, bxq_mark, byq_mark, byq_cycle
            )
        loss_q = self.A.get_loss(outputs_q, batch_y_q, params=fast_params)
        return np.mean(support_losses), loss_q

    def meta_validate(self, vali_loader):
        """
        在 meta 阶段对 A 进行验证（冻结 base model），统计 query 损失形式。
        """
        self.A.eval()
        total_cov_loss = []
        criterion = nn.MSELoss()
        total_mse = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in vali_loader:
                outputs, batch_y, _ = self.forward_step(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle
                )
                cov_loss = self.A.get_loss(outputs, batch_y)
                mse_loss = criterion(outputs, batch_y)
                total_cov_loss.append(cov_loss)
                total_mse.append(mse_loss)

        mean_cov = torch.mean(torch.stack(total_cov_loss)).item()
        mean_mse = torch.mean(torch.stack(total_mse)).item()
        self.A.train()
        return mean_cov, mean_mse

    def meta_train(self, train_data, vali_loader, setting):
        """
        Phase 1: 只训练 A
        """
        print("\n================= Phase 1: Meta Learning A =================\n")
        # 冻结 base model
        disable_grad(self.model)
        self.model.eval()

        enable_grad(self.A)
        self.A.train()

        support_loader_list, query_loader_list = self.initialize_meta_tasks(train_data)

        # 优化器 & early stopping
        A_optim = self._select_optimizer(self.A, self.meta_lr, optim_type=getattr(self.args, 'meta_optim_type', 'adam'))
        A_scheduler = Scheduler(A_optim, self.args, steps_per_epoch=len(train_data) if hasattr(train_data, '__len__') else 100)
        meta_early_stopping = EarlyStopping(patience=self.meta_patience, verbose=True, prefix='[Meta]')

        # 准备日志目录
        meta_ckpt_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(meta_ckpt_dir, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)

        if self.writer is None:
            self.writer = self._create_writer(res_path)

        best_metric = None

        for epoch in range(1, self.meta_epochs + 1):
            epoch_support_losses, epoch_query_losses = [], []
            t0 = time.time()

            for task_id, (support_loader, query_loader) in enumerate(zip(support_loader_list, query_loader_list)):
                # 一个 outer step: 对单任务执行 inner_loop
                A_optim.zero_grad()
                support_l, query_l = self.inner_loop(support_loader, query_loader)
                query_l.backward()
                A_optim.step()

                self.meta_global_step += 1
                epoch_support_losses.append(support_l)
                epoch_query_losses.append(query_l.item())

                if task_id % 20 == 0:
                    print(f"[Meta][Epoch {epoch}] Task {task_id} | Support: {support_l:.5f} | Query: {query_l.item():.5f}")

            # 调度器（可根据策略自行调整，这里如果 lradj == 'TST' 按 step，否则按 epoch 验证）
            if getattr(self.args, 'lradj', None) in ['TST']:
                A_scheduler.step(verbose=False)

            mean_support = np.mean(epoch_support_losses)
            mean_query = np.mean(epoch_query_losses)

            # 验证
            vali_cov, vali_mse = self.meta_validate(vali_loader)

            self.writer.add_scalar(f'{self.pred_len}/meta/support_loss', mean_support, epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta/query_loss', mean_query, epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta/vali_cov', vali_cov, epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta/vali_mse', vali_mse, epoch)
            log_heatmap(self.writer, get_projection(self.A), f'{self.pred_len}/meta/cov_mat', epoch)

            print(f"[Meta][Epoch {epoch}] Time: {time.time() - t0:.2f}s | "
                  f"Train Support: {mean_support:.5f} | Train Query: {mean_query:.5f} | "
                  f"Vali Cov: {vali_cov:.5f} | Vali MSE: {vali_mse:.5f}")

            # 用验证 cov 或 query 之一做 early stopping，这里选择 vali_cov
            metric_to_track = vali_cov
            improved = meta_early_stopping(metric_to_track, self.A, meta_ckpt_dir, file_name='A_meta.pth')
            if improved:
                best_metric = metric_to_track

            if meta_early_stopping.early_stop:
                print("[Meta] Early stopping triggered.")
                break

            if getattr(self.args, 'lradj', None) not in ['TST']:
                A_scheduler.step(metric_to_track, epoch)

        # 加载最优 A
        best_A_path = os.path.join(meta_ckpt_dir, 'A_meta.pth')
        if os.path.exists(best_A_path):
            self.A = torch.load(best_A_path)
        else:
            torch.save(self.A, best_A_path)
        self.meta_trained = True
        print(f"[Meta] Finished. Best metric (vali cov) = {best_metric}")

    # =============== Phase 2: Train Base Model with Fixed A ===============

    def vali(self, vali_data, vali_loader, criterion):
        total_loss_cov = []
        total_loss_mse = []
        self.model.eval()
        self.A.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in vali_loader:
                outputs, batch_y, _ = self.forward_step(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle
                )
                loss_cov = self.A.get_loss(outputs, batch_y)
                loss_mse = criterion(outputs, batch_y)
                total_loss_cov.append(loss_cov)
                total_loss_mse.append(loss_mse)

        cov_mean = torch.mean(torch.stack(total_loss_cov)).item()
        mse_mean = torch.mean(torch.stack(total_loss_mse)).item()

        self.model.train()
        self.A.train()  # 虽然后面不更新 A，但保持接口一致
        return mse_mean, cov_mean

    def base_train(self, train_loader, vali_data, vali_loader, setting):
        """
        Phase 2: 训练 base model, A 固定。
        """
        print("\n================= Phase 2: Train Base Model with Fixed A =================\n")
        # 固定 A
        disable_grad(self.A)
        self.A.eval()

        enable_grad(self.model)
        self.model.train()

        # 路径
        ckpt_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)

        if self.writer is None:
            self.writer = self._create_writer(res_path)

        model_optim = self._select_optimizer(self.model, self.lr)
        scheduler = Scheduler(model_optim, self.args, len(train_loader))
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        criterion = self._select_criterion()

        for epoch in range(1, self.train_epochs + 1):
            t0 = time.time()
            self.model.train()
            iter_losses_cov, iter_losses_mse = [], []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                model_optim.zero_grad()
                outputs, batch_y, _ = self.forward_step(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle
                )
                loss_cov = self.A.get_loss(outputs, batch_y)
                loss_cov.backward()
                model_optim.step()

                with torch.no_grad():
                    loss_mse = criterion(outputs, batch_y)

                iter_losses_cov.append(loss_cov.item())
                iter_losses_mse.append(loss_mse.item())

                self.train_global_step += 1
                if self.train_global_step % 100 == 0:
                    print(f"[Train][Epoch {epoch}] Iter {i+1}/{len(train_loader)} "
                          f"| CovLoss: {loss_cov.item():.5f} | MSE: {loss_mse.item():.5f}")

            # 学习率调度（逐 step 策略）
            if getattr(self.args, 'lradj', None) in ['TST']:
                scheduler.step(verbose=False)

            mean_cov = np.mean(iter_losses_cov)
            mean_mse = np.mean(iter_losses_mse)

            vali_mse, vali_cov = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss_cov', mean_cov, epoch)
            self.writer.add_scalar(f'{self.pred_len}/train/loss_mse', mean_mse, epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_cov', vali_cov, epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_mse', vali_mse, epoch)
            log_heatmap(self.writer, get_projection(self.A), f'{self.pred_len}/base/cov_mat', epoch)

            print(f"[Base][Epoch {epoch}] Time: {time.time() - t0:.2f}s | "
                  f"Train Cov: {mean_cov:.5f}, Train MSE: {mean_mse:.5f} | "
                  f"Vali Cov: {vali_cov:.5f}, Vali MSE: {vali_mse:.5f}")

            other_to_save = {'A_meta_fixed': self.A}
            improved = early_stopping(vali_mse, self.model, ckpt_dir, **other_to_save)
            if early_stopping.early_stop:
                print("[Base] Early stopping.")
                break

            if getattr(self.args, 'lradj', None) not in ['TST']:
                scheduler.step(vali_mse, epoch)

        # 加载最优
        best_model_path = os.path.join(ckpt_dir, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        best_A_path = os.path.join(ckpt_dir, 'A_meta_fixed.pth')
        if os.path.exists(best_A_path):
            # 保证 test 时读取
            self.A = torch.load(best_A_path)

    def train(self, setting, prof=None):
        """
        总训练入口：Phase 1 (meta) -> Phase 2 (base)
        """
        # 取数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # Phase 1
        if not self.meta_trained:
            self.meta_train(train_data, vali_loader, setting)

        # Phase 2
        self.base_train(train_loader, vali_data, vali_loader, setting)

        return self.model

    # =============== Testing ===============

    def test(self, setting, prof=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        ckpt_dir = os.path.join(self.args.checkpoints, setting)

        if test:
            print('[Test] Loading model & A ...')
            model_path = os.path.join(ckpt_dir, 'checkpoint.pth')
            A_path_meta = os.path.join(ckpt_dir, 'A_meta_fixed.pth')
            if not os.path.exists(A_path_meta):
                A_path_meta = os.path.join(ckpt_dir, 'A_meta.pth')
            self.model.load_state_dict(torch.load(model_path))
            self.A = torch.load(A_path_meta)

        self.model.eval()
        self.A.eval()

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                outputs, batch_y, _ = self.forward_step(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle
                )
                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and getattr(self.args, 'inverse', False):
                    # inverse transform
                    bx_np = batch_x.cpu().numpy()
                    in_shape = bx_np.shape
                    bx_np = test_data.inverse_transform(bx_np.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(bx_np).float().to(self.device)

                    out_shape = outputs.shape
                    out_np = outputs.cpu().numpy()
                    by_np = batch_y.cpu().numpy()
                    out_np = test_data.inverse_transform(out_np.reshape(-1, out_shape[-1])).reshape(out_shape)
                    by_np = test_data.inverse_transform(by_np.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(out_np).float().to(self.device)
                    batch_y = torch.from_numpy(by_np).float().to(self.device)

                inputs.append(batch_x.cpu())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate(
                        (batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()),
                        axis=0
                    )
                    pd = np.concatenate(
                        (batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()),
                        axis=0
                    )
                    visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        with torch.no_grad():
            self.A.to(preds.device)
            cov_loss = self.A.get_loss(preds, trues)

        print(f'[Test][{self.pred_len}] mse:{mse}, mae:{mae}, cov:{cov_loss}')

        # TensorBoard
        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, getattr(self, 'epoch', 0))
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, getattr(self, 'epoch', 0))
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, getattr(self, 'epoch', 0))
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, getattr(self, 'epoch', 0))
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, getattr(self, 'epoch', 0))
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, getattr(self, 'epoch', 0))
        self.writer.add_scalar(f'{self.pred_len}/test/cov', cov_loss, getattr(self, 'epoch', 0))
        self.writer.close()

        log_path = getattr(self.args, 'log_path', "result_long_term_forecast.txt")
        with open(log_path, 'a') as f:
            f.write(setting + "\n")
            f.write(f'mse:{mse}, mae:{mae}, cov:{cov_loss}\n\n')

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, cov_loss.item(), rmse, mape, mspe, mre]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())

        # 保存配置
        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return