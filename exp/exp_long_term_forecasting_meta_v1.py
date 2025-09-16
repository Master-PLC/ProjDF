import os
import time
import warnings
from itertools import cycle
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import yaml
from exp.exp_basic import Exp_Basic
from torch.utils.data import DataLoader
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.tools import EarlyStopping, Scheduler, clip_grads, disable_grad, enable_grad, log_heatmap, split_dataset, \
    split_dataset_with_overlap, visual

warnings.filterwarnings('ignore')


class CovarianceMatrix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.L_param = nn.Parameter(torch.eye(args.pred_len))
        self.eps = 1e-6
        self.auxi_loss = args.auxi_loss

    def _get_L(self, params=None):
        if params is None:
            L_param = self.L_param
        else:
            L_param = params['L_param']

        # 取下三角并在对角线加 eps，确保正定
        L = torch.tril(L_param)
        diag = torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps)
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + diag
        return L

    def forward(self, params=None):
        L = self._get_L(params)
        return L @ L.transpose(-1, -2)               # Σ = L Lᵀ

    def get_inverse(self, params=None):
        L = self._get_L(params)
        A = L @ L.transpose(-1, -2)
        return torch.linalg.inv(A)

    def get_loss(self, pred, target, params=None):
        """ML3 Learned Loss Function"""
        L = self._get_L(params)  # [P, P] 下三角矩阵

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
            Sigma = self.forward(params)  # [P, P]
            off_diag = Sigma - torch.diag_embed(torch.diagonal(Sigma))
            reg_loss = torch.norm(off_diag, p='fro') ** 2
            loss += self.args.reg_lambda * reg_loss

        return loss


def get_projection(A):
    with torch.no_grad():
        Am = A()
    return Am.detach().cpu().numpy()  # 返回 numpy 数组


def get_param_dict(module):
    # 返回 OrderedDict，适用于 functional forward
    return dict(module.named_parameters())


def update_param_dict(param_dict, grads, lr):
    # param_dict: dict key->tensor, grads: dict key->grad
    return {k: v - lr * grads[k] for k, v in param_dict.items()}


class Exp_Long_Term_Forecast_ML3(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.n_inner = args.meta_inner_steps
        self.lr = args.learning_rate
        self.inner_lr = args.inner_lr
        self.meta_lr = args.meta_lr
        self.first_order = args.first_order

        self.A = CovarianceMatrix(self.args).to(self.device)
        # 保存初始模型状态用于meta test phase重新初始化
        self.initial_model_state = None

    def save_initial_model_state(self):
        """保存模型的初始状态"""
        self.initial_model_state = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def reset_model_to_initial_state(self):
        """将模型重置为初始状态"""
        if self.initial_model_state is not None:
            self.model.load_state_dict(self.initial_model_state)
            print("Model reset to initial state for meta test phase")
        else:
            print("Warning: No initial model state saved, using current state")

    @contextmanager
    def temp_model_params(self, param_dict):
        """
        临时替换模型参数的上下文管理器
        使用方法：
        with self.temp_model_params(new_params):
            output = self.model(input)
        """
        # 保存当前参数
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        # 设置新参数
        for name, param in self.model.named_parameters():
            if name in param_dict:
                param.data = param_dict[name]
        
        try:
            yield
        finally:
            # 恢复原始参数
            for name, param in self.model.named_parameters():
                param.data = original_params[name]

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

                loss = criterion(pred, true)  # 标准损失
                cov_loss = self.A.get_loss(pred, true)  # 学习到的损失

                total_loss.append(loss)
                total_cov_loss.append(cov_loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        total_loss = torch.mean(torch.stack(total_loss)).item()
        total_cov_loss = torch.mean(torch.stack(total_cov_loss)).item()

        self.model.train()
        self.A.train()
        return total_loss, total_cov_loss

    def inner_loop(self, task_id, support_loader, query_loader):
        # 获取当前模型参数（每个meta epoch都从当前状态开始，而不是初始状态）
        model_params_init = get_param_dict(self.model)

        # 内层循环：使用学习到的损失函数训练模型参数
        fast_model_params = {k: v.clone() for k, v in model_params_init.items()}
        for k in range(self.n_inner):
            bx, by, bx_mark, by_mark, by_cycle = next(support_loader)
            outputs, batch_y, _ = self.forward_step_with_params(
                bx, by, bx_mark, by_mark, by_cycle, fast_model_params
            )
            loss = self.A.get_loss(outputs, batch_y)

            # 计算模型参数的梯度
            model_grads = torch.autograd.grad(
                loss, fast_model_params.values(), 
                create_graph=not self.first_order, 
                allow_unused=True
            )
            model_grads = clip_grads(model_grads, self.args.max_norm)
            model_grads_dict = {k: g for k, g in zip(fast_model_params.keys(), model_grads) if g is not None}
            fast_model_params = update_param_dict(fast_model_params, model_grads_dict, self.inner_lr)

        # 外层循环：在query set上使用标准损失评估性能
        bx, by, bx_mark, by_mark, by_cycle = next(query_loader)
        outputs, batch_y, _ = self.forward_step_with_params(
            bx, by, bx_mark, by_mark, by_cycle, fast_model_params
        )
        # 使用标准损失（如MSE）作为元目标
        meta_loss = nn.MSELoss()(outputs, batch_y)
        return meta_loss

    def forward_step_with_params(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, params):
        # 使用临时参数进行前向传播
        with self.temp_model_params(params):
            outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
        return outputs, batch_y, None

    def initialize_meta_tasks(self, train_data):
        self.meta_learning = False

        task_data_list = split_dataset_with_overlap(train_data, self.args.num_tasks, self.args.overlap_ratio)
        task_data_list = [split_dataset(task_data, r=0.7) for task_data in task_data_list]

        support_data_list = [td[0] for td in task_data_list]
        support_loader_list = [DataLoader(support_data, batch_size=self.args.auxi_batch_size, shuffle=True) for support_data in support_data_list]
        support_loader_list = [cycle(support_loader) for support_loader in support_loader_list]

        query_data_list = [td[1] for td in task_data_list]
        query_loader_list = [DataLoader(query_data, batch_size=self.args.auxi_batch_size, shuffle=True) for query_data in query_data_list]
        query_loader_list = [cycle(query_loader) for query_loader in query_loader_list]
        return support_loader_list, query_loader_list

    def meta_train(self, support_loader_list, query_loader_list, criterion, path, res_path):
        # 在meta train阶段，损失函数参数可训练，模型参数也需要梯度（用于inner loop）
        enable_grad(self.A)
        enable_grad(self.model)

        A_optim = self._select_optimizer(self.A, self.meta_lr, optim_type=getattr(self.args, 'meta_optim_type', 'Adam'))
        A_scheduler = Scheduler(A_optim, self.args, self.args.warmup_epochs)

        time_now = time.time()
        meta_epoch = 0
        for epoch in range(self.args.warmup_epochs):
            meta_epoch = epoch + 1
            total_meta_loss = 0
            task_losses = []

            meta_lr_cur = A_scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_lr', meta_lr_cur, self.epoch)

            self.model.train()
            self.A.train()

            A_optim.zero_grad()
            epoch_time = time.time()
            # 遍历所有任务，累积meta loss
            for task_id, (support_loader, query_loader) in enumerate(zip(support_loader_list, query_loader_list)):
                meta_loss = self.ml3_task_meta_loss(task_id, support_loader, query_loader)
                
                if meta_loss is not None:
                    total_meta_loss += meta_loss
                    valid_tasks += 1
                    task_losses.append(meta_loss.item())
                    
                    if (task_id + 1) % 10 == 0:
                        print(f"\tMeta Train - Task: {task_id + 1}/{len(support_loader_list)}, Epoch: {self.epoch} | Task Meta Loss: {meta_loss.item():.7f}")
                
                self.step += 1
            
            # 统一进行损失函数参数的更新
            if valid_tasks > 0:
                avg_meta_loss = total_meta_loss / valid_tasks
                
                # 反向传播并更新损失函数参数
                avg_meta_loss.backward()
                
                # 梯度裁剪（可选）
                if hasattr(self.args, 'max_norm') and self.args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.A.parameters(), self.args.max_norm)
                
                # 更新损失函数参数
                A_optim.step()
                
                avg_meta_loss_val = avg_meta_loss.item()
            else:
                avg_meta_loss_val = 0.0
            
            # 记录日志
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_loss', avg_meta_loss_val, self.epoch)
            log_heatmap(self.writer, get_projection(self.A), f'{self.pred_len}/meta_train_loss_matrix', self.epoch)
            
            print(f"Meta Train Epoch: {self.epoch} | Avg Meta Loss: {avg_meta_loss_val:.7f} ({valid_tasks} tasks)")
            print(f"Meta Train Epoch: {self.epoch} cost time: {time.time() - epoch_time:.2f}s")
            
            if self.args.lradj not in ['TST']:
                A_scheduler.step(avg_meta_loss_val, self.epoch)
            else:
                A_scheduler.step(verbose=True)
        
        # 保存学习到的损失函数
        best_loss_path = os.path.join(path, 'cov_loss.pth')
        torch.save(self.A, best_loss_path)
        print(f"Saved learned loss function to {best_loss_path}")
        
        print(f"\nMeta Train Phase completed!")
        print("Learned loss function is ready for meta test phase")

    def meta_test_phase(self, train_loader, vali_data, vali_loader, criterion, path, res_path):
        """ML3 Meta Test 阶段：重新初始化模型，使用学习到的损失函数训练模型"""
        print(f"\n{'='*50}")
        print(f"Starting ML3 Meta Test Phase for {self.test_epochs} epochs")
        print(f"Model will be reset to initial state")
        print(f"Using learned loss function to train the model")
        print(f"{'='*50}\n")
        
        # 关键步骤：重新初始化模型到初始状态
        self.reset_model_to_initial_state()
        
        # 固定损失函数参数，训练模型参数
        disable_grad(self.A)
        enable_grad(self.model)
        
        model_optim = self._select_optimizer(self.model, self.lr)
        scheduler = Scheduler(model_optim, self.args, len(train_loader))
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        time_now = time.time()
        
        for epoch in range(self.test_epochs):
            self.epoch = self.meta_epochs + epoch + 1  # 继续epoch计数
            iter_count = 0
            train_loss_learned, train_loss_mse = [], []
            
            lr_cur = scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_test/lr', lr_cur, self.epoch)
            
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.model.train()
                self.A.eval()
                
                iter_count += 1
                model_optim.zero_grad()
                
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                
                # 使用学习到的损失函数训练模型
                loss_learned = self.A.get_loss(outputs, batch_y)
                loss_learned.backward()
                
                # 梯度裁剪（可选）
                if hasattr(self.args, 'max_norm') and self.args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                
                model_optim.step()
                
                # 记录标准MSE损失用于监控
                with torch.no_grad():
                    loss_mse = criterion(outputs, batch_y)
                
                train_loss_learned.append(loss_learned.item())
                train_loss_mse.append(loss_mse.item())
                
                self.step += 1
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_learned', loss_learned.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_mse', loss_mse.item(), self.step)
                
                if (i + 1) % 100 == 0:
                    print(f"\tMeta Test - iters: {i + 1}, epoch: {self.epoch} | learned loss: {loss_learned.item():.7f}, mse loss: {loss_mse.item():.7f}")
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.test_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; cost time: {cost_time:.4f}s; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == len(train_loader)))
            
            avg_train_loss_learned = np.mean(train_loss_learned)
            avg_train_loss_mse = np.mean(train_loss_mse)
            
            # 在meta test阶段进行validation
            valid_loss_mse, valid_loss_learned = self.vali(vali_data, vali_loader, criterion)
            
            self.writer.add_scalar(f'{self.pred_len}/meta_test/train_loss_learned', avg_train_loss_learned, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/train_loss_mse', avg_train_loss_mse, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/valid_loss_learned', valid_loss_learned, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/valid_loss_mse', valid_loss_mse, self.epoch)
            log_heatmap(self.writer, get_projection(self.A), f'{self.pred_len}/meta_test_loss_matrix', self.epoch)
            
            print(f"Meta Test Epoch: {self.epoch} | Train Learned: {avg_train_loss_learned:.7f}, MSE: {avg_train_loss_mse:.7f} | Valid Learned: {valid_loss_learned:.7f}, MSE: {valid_loss_mse:.7f}")
            print(f"Meta Test Epoch: {self.epoch} cost time: {time.time() - epoch_time:.2f}s")
            
            # 在meta test阶段使用early stopping保存最佳模型
            other_to_save = {'cov_loss': self.A}
            improved = early_stopping(valid_loss_mse, self.model, path, **other_to_save)
            
            if early_stopping.early_stop:
                print("Meta Test Early stopping")
                break
                
            if self.args.lradj not in ['TST']:
                scheduler.step(valid_loss_mse, self.epoch)
        
        print(f"\nMeta Test Phase completed! Best validation MSE: {early_stopping.best_score:.7f}")

    def train(self, setting, prof=None):
        train_data, train_loader = self._get_data(flag='train')
        support_loader_list, query_loader_list = self.initialize_meta_tasks(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        criterion = self._select_criterion()
        self.save_initial_model_state()

        # ============ Meta Train 阶段：只训练损失函数 ============
        self.meta_train(support_loader_list, query_loader_list, criterion, path, res_path)
        # ============ ML3 Meta Test 阶段：重新初始化模型，使用学习到的损失函数训练 ============
        self.meta_test_phase(train_loader, vali_data, vali_loader, criterion, path, res_path)
        
        # 加载最佳模型和损失函数
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model from meta test phase")
        
        best_loss_path = os.path.join(path, 'cov_loss.pth')
        if os.path.exists(best_loss_path):
            self.A = torch.load(best_loss_path)
            print("Loaded best learned loss function")

        return self.model

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