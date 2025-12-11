import os
import time
import torch
import warnings
import yaml

from collections import OrderedDict
from itertools import cycle
import numpy as np
from torch.func import functional_call
import torch.nn as nn
from torch.utils.data import DataLoader

from exp.exp_basic import Exp_Basic
from models import MODEL_REQUIRES_CYCLE
from utils.metrics_torch import metric_torch
from utils.tools import EarlyStopping, Scheduler, clip_grads, disable_grad, enable_grad, log_heatmap, plot_heatmap, split_dataset, split_dataset_with_overlap, visual

warnings.filterwarnings('ignore')


class CovarianceMatrix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.eps = 1e-6
        self.auxi_loss = args.auxi_loss
        self.meta_type = getattr(args, 'meta_type', 'all')

        if self.meta_type == 'all':
            self.L_param = nn.Parameter(torch.eye(args.pred_len))
        elif self.meta_type == 'diag':
            self.diag_param = nn.Parameter(torch.ones(args.pred_len))
        elif self.meta_type == 'off_diag':
            self.L_param = nn.Parameter(torch.zeros(args.pred_len, args.pred_len))
        else:
            raise ValueError(f"Unknown meta_type: {self.meta_type}. Supported types: ['all', 'diag', 'off_diag']")

    def _get_L(self, params=None):
        if self.meta_type == 'all':
            # 原始模式：完整的Cholesky分解
            if params is None:
                L_param = self.L_param
            else:
                L_param = params['L_param']
            
            # 取下三角并在对角线加 eps，确保正定
            L = torch.tril(L_param)
            diag = torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps)
            L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + diag
            return L
            
        elif self.meta_type == 'diag':
            # 对角线模式：L矩阵是对角矩阵
            if params is None:
                diag_param = self.diag_param
            else:
                diag_param = params['diag_param']
            
            # 对角矩阵的L就是对角线元素的平方根，确保正值
            diag_values = torch.sqrt(torch.abs(diag_param) + self.eps)
            return torch.diag(diag_values)
            
        elif self.meta_type == 'off_diag':
            # 非对角线模式：对角线固定为1，只学习下三角非对角线部分
            if params is None:
                L_param = self.L_param
            else:
                L_param = params['L_param']
            
            # 只取严格下三角部分（不包括对角线），对角线固定为1
            L = torch.tril(L_param, diagonal=-1)
            L = L + torch.eye(self.pred_len, device=L.device, dtype=L.dtype)
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

        if self.meta_type == 'diag':
            # 对于对角矩阵，可以直接进行元素级除法
            diag_inv = 1.0 / torch.diag(L)  # L的对角线元素的倒数
            x = E_flat * diag_inv.unsqueeze(0)  # 广播乘法
        else:
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

        if self.args.reg_lambda > 0 and self.meta_type in ['all', 'off_diag']:
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


def _list_dot(xs, ys):
    return sum((x * y).sum() for x, y in zip(xs, ys))


def _list_add(xs, ys, alpha=1.0):
    return [x + alpha * y for x, y in zip(xs, ys)]


def _list_sub(xs, ys):
    return [x - y for x, y in zip(xs, ys)]


def _list_mul(xs, scalar):
    return [x * scalar for x in xs]


def _zeros_like_list(xs):
    return [torch.zeros_like(x) for x in xs]


def _replace_none_grads(params, grads):
    out = []
    for p, g in zip(params, grads):
        if g is None:
            out.append(torch.zeros_like(p))
        else:
            out.append(g)
    return out


def conjugate_gradient(hvp_fn, b_list, iters=10, tol=1e-10, damping=1e-3):
    """改进的共轭梯度法，增加数值稳定性"""
    x = _zeros_like_list(b_list)
    r = [b.clone() for b in b_list]
    p = [ri.clone() for ri in r]
    rdotr = _list_dot(r, r)
    
    # 添加数值检查
    if torch.isnan(rdotr) or rdotr < 1e-16:
        print("Warning: Initial residual is NaN or too small in CG")
        return _zeros_like_list(b_list)

    for i in range(iters):
        try:
            Ap = hvp_fn(p)  # list
            
            # 检查HVP结果
            if any(torch.isnan(ap).any() or torch.isinf(ap).any() for ap in Ap):
                print(f"Warning: NaN/Inf in HVP at CG iteration {i}")
                break
                
            pAp = _list_dot(p, Ap)
            # 添加阻尼项防止除零
            denom = pAp + damping
            
            if abs(denom) < 1e-16:
                print(f"Warning: Denominator too small in CG iteration {i}")
                break
                
            alpha = rdotr / denom
            
            # 检查alpha是否合理
            if torch.isnan(alpha) or torch.isinf(alpha) or abs(alpha) > 1e6:
                print(f"Warning: Invalid alpha in CG iteration {i}: {alpha}")
                break
                
            x = [xi + alpha * pi for xi, pi in zip(x, p)]
            r = [ri - alpha * Api for ri, Api in zip(r, Ap)]
            
            new_rdotr = _list_dot(r, r)
            
            # 检查收敛条件和数值稳定性
            if torch.isnan(new_rdotr) or new_rdotr < 1e-16:
                print(f"Warning: Invalid residual in CG iteration {i}")
                break
                
            if torch.sqrt(new_rdotr) < tol:
                break
                
            beta = new_rdotr / (rdotr + 1e-16)
            
            if torch.isnan(beta) or torch.isinf(beta):
                print(f"Warning: Invalid beta in CG iteration {i}")
                break
                
            p = [ri + beta * pi for ri, pi in zip(r, p)]
            rdotr = new_rdotr
            
        except Exception as e:
            print(f"Error in CG iteration {i}: {e}")
            break
    
    # 最终检查结果
    if any(torch.isnan(xi).any() or torch.isinf(xi).any() for xi in x):
        print("Warning: NaN/Inf in final CG result, returning zeros")
        return _zeros_like_list(b_list)
        
    return x


class Exp_Long_Term_Forecast_META_iMAML(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.n_inner = args.meta_inner_steps
        self.lr = args.learning_rate
        self.inner_lr = args.inner_lr
        self.meta_lr = args.meta_lr
        self.first_order = args.first_order
        self.model_per_task = args.model_per_task
        self.num_tasks = args.num_tasks

        # iMAML 专用超参（若 args 未配置则使用默认值）
        self.implicit_lambda = getattr(args, 'implicit_lambda', 1e-3)  # proximal 系数 λ
        self.cg_iters = getattr(args, 'cg_iters', 10)
        self.cg_tol = getattr(args, 'cg_tol', 1e-10)
        self.cg_damping = getattr(args, 'cg_damping', 0.0)  # 额外 damping，可设为 0

        self.A = CovarianceMatrix(self.args).to(self.device)
        self.task_models = [self.model]
        if self.model_per_task:
            for _ in range(1, self.num_tasks):
                task_model = self._build_model().to(self.device)
                self.task_models.append(task_model)
        else:
            self.task_models = [self.model] * self.num_tasks

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

    def _inner_adapt_with_prox(self, task_model, fast_params, init_params, support_batch):
        """
        在支持集上做 K 步近端 SGD：min_θ L_support(θ, A) + (λ/2)||θ - θ0||^2
        fast_params/init_params: dict[name: tensor]，其 tensor 均 requires_grad=True
        """
        bx, by, bx_mark, by_mark, by_cycle = support_batch
        for k in range(self.n_inner):
            outputs, batch_y, _ = self.forward_step_with_params(
                bx, by, bx_mark, by_mark, by_cycle, fast_params, task_model
            )
            loss_support = self.A.get_loss(outputs, batch_y)
            prox = 0.0
            for name in fast_params.keys():
                prox = prox + 0.5 * self.implicit_lambda * torch.sum((fast_params[name] - init_params[name]) ** 2)
            f_obj = loss_support + prox

            grads = torch.autograd.grad(
                f_obj, list(fast_params.values()),
                create_graph=False, retain_graph=False, allow_unused=True
            )
            grads = _replace_none_grads(list(fast_params.values()), grads)
            # SGD update
            for (name, param), g in zip(fast_params.items(), grads):
                fast_params[name] = param - self.inner_lr * g
        return fast_params

    def _hvp_fn(self, task_model, theta_params, init_params, support_batch):
        """改进的HVP函数，增加数值稳定性"""
        bx, by, bx_mark, by_mark, by_cycle = support_batch

        def hvp(v_list):
            try:
                # 检查输入v_list
                if any(torch.isnan(v).any() or torch.isinf(v).any() for v in v_list):
                    print("Warning: NaN/Inf in HVP input v_list")
                    return _zeros_like_list(v_list)
                
                # 重新计算目标函数
                outputs, batch_y, _ = self.forward_step_with_params(
                    bx, by, bx_mark, by_mark, by_cycle, theta_params, task_model
                )
                
                # 检查前向传播结果
                if torch.isnan(outputs).any() or torch.isnan(batch_y).any():
                    print("Warning: NaN in forward pass during HVP")
                    return _zeros_like_list(v_list)
                
                loss_support = self.A.get_loss(outputs, batch_y)
                
                # 检查支持集损失
                if torch.isnan(loss_support) or torch.isinf(loss_support):
                    print("Warning: NaN/Inf in support loss during HVP")
                    return _zeros_like_list(v_list)
                
                # 近端正则项
                prox = 0.0
                for name in theta_params.keys():
                    diff = theta_params[name] - init_params[name]
                    prox = prox + 0.5 * self.implicit_lambda * torch.sum(diff ** 2)
                
                f_obj = loss_support + prox
                
                # 检查目标函数
                if torch.isnan(f_obj) or torch.isinf(f_obj):
                    print("Warning: NaN/Inf in objective during HVP")
                    return _zeros_like_list(v_list)

                # 计算一阶梯度
                theta_list = list(theta_params.values())
                g_theta = torch.autograd.grad(
                    f_obj, theta_list,
                    create_graph=True, retain_graph=True, allow_unused=True
                )
                g_theta = _replace_none_grads(theta_list, g_theta)
                
                # 检查一阶梯度
                if any(torch.isnan(g).any() or torch.isinf(g).any() for g in g_theta):
                    print("Warning: NaN/Inf in first-order gradients during HVP")
                    return _zeros_like_list(v_list)

                # 计算梯度与v的内积
                dot = _list_dot(g_theta, v_list)
                
                if torch.isnan(dot) or torch.isinf(dot):
                    print("Warning: NaN/Inf in dot product during HVP")
                    return _zeros_like_list(v_list)

                # 计算二阶梯度（HVP）
                hv = torch.autograd.grad(
                    dot, theta_list,
                    retain_graph=True, allow_unused=True
                )
                hv = _replace_none_grads(theta_list, hv)

                # 检查HVP结果
                if any(torch.isnan(h).any() or torch.isinf(h).any() for h in hv):
                    print("Warning: NaN/Inf in HVP result")
                    return _zeros_like_list(v_list)

                # 添加阻尼
                if self.cg_damping > 0:
                    hv = [hvi + self.cg_damping * vi for hvi, vi in zip(hv, v_list)]

                return hv
                
            except Exception as e:
                print(f"Error in HVP computation: {e}")
                return _zeros_like_list(v_list)

        return hvp

    def _implicit_meta_grad_A(self, task_model, init_params, theta_params, support_batch, query_batch):
        """改进的隐式梯度计算，增加数值稳定性检查"""
        try:
            # 1) 计算验证集损失和梯度
            bx_q, by_q, bxm_q, bym_q, cyc_q = query_batch
            outputs_q, batch_y_q, _ = self.forward_step_with_params(
                bx_q, by_q, bxm_q, bym_q, cyc_q, theta_params, task_model
            )
            
            # 检查前向传播结果
            if torch.isnan(outputs_q).any() or torch.isnan(batch_y_q).any():
                print("Warning: NaN in query forward pass")
                return self._get_zero_grads_A(), float('inf')
            
            L_val = self.A.get_loss(outputs_q, batch_y_q)
            
            # 检查验证损失
            if torch.isnan(L_val) or torch.isinf(L_val):
                print("Warning: NaN/Inf in validation loss")
                return self._get_zero_grads_A(), float('inf')

            theta_list = list(theta_params.values())
            g_val_theta = torch.autograd.grad(
                L_val, theta_list, retain_graph=True, allow_unused=True
            )
            g_val_theta = _replace_none_grads(theta_list, g_val_theta)
            
            # 检查验证集梯度
            if any(torch.isnan(g).any() or torch.isinf(g).any() for g in g_val_theta):
                print("Warning: NaN/Inf in validation gradients")
                return self._get_zero_grads_A(), float('inf')

            # 2) 使用共轭梯度求解
            hvp = self._hvp_fn(task_model, theta_params, init_params, support_batch)
            s_list = conjugate_gradient(
                hvp, g_val_theta, 
                iters=self.cg_iters, 
                tol=self.cg_tol,
                damping=max(self.cg_damping, 1e-4)  # 确保有最小阻尼
            )
            
            # 检查CG结果
            if any(torch.isnan(s).any() or torch.isinf(s).any() for s in s_list):
                print("Warning: NaN/Inf in CG solution")
                return self._get_zero_grads_A(), float('inf')
            
            s_list_detached = [s.detach() for s in s_list]

            # 3) 计算第一项：∇_A L_val
            A_params = list(self.A.parameters())
            try:
                g_val_A = torch.autograd.grad(
                    L_val, A_params, retain_graph=True, allow_unused=True
                )
                g_val_A = [torch.zeros_like(p) if g is None else g for p, g in zip(A_params, g_val_A)]
                
                # 检查第一项梯度
                if any(torch.isnan(g).any() or torch.isinf(g).any() for g in g_val_A):
                    print("Warning: NaN/Inf in first term gradient wrt A")
                    return self._get_zero_grads_A(), float('inf')
                    
            except Exception as e:
                print(f"Error computing first term: {e}")
                return self._get_zero_grads_A(), float('inf')

            # 4) 计算第二项：∂/∂A [ ∇_θ f(θ*, A) ⋅ s ]
            try:
                bx_s, by_s, bxm_s, bym_s, cyc_s = support_batch
                outputs_s, batch_y_s, _ = self.forward_step_with_params(
                    bx_s, by_s, bxm_s, bym_s, cyc_s, theta_params, task_model
                )
                
                if torch.isnan(outputs_s).any() or torch.isnan(batch_y_s).any():
                    print("Warning: NaN in support forward pass for second term")
                    return self._get_zero_grads_A(), float('inf')
                
                L_sup = self.A.get_loss(outputs_s, batch_y_s)
                
                if torch.isnan(L_sup) or torch.isinf(L_sup):
                    print("Warning: NaN/Inf in support loss for second term")
                    return self._get_zero_grads_A(), float('inf')
                
                # 近端项
                prox = 0.0
                for name in theta_params.keys():
                    diff = theta_params[name] - init_params[name]
                    prox = prox + 0.5 * self.implicit_lambda * torch.sum(diff ** 2)
                
                f_obj = L_sup + prox

                g_sup_theta = torch.autograd.grad(
                    f_obj, theta_list, create_graph=True, retain_graph=True, allow_unused=True
                )
                g_sup_theta = _replace_none_grads(theta_list, g_sup_theta)
                
                if any(torch.isnan(g).any() or torch.isinf(g).any() for g in g_sup_theta):
                    print("Warning: NaN/Inf in support gradients for second term")
                    return self._get_zero_grads_A(), float('inf')

                dot_cross = _list_dot(g_sup_theta, s_list_detached)
                
                if torch.isnan(dot_cross) or torch.isinf(dot_cross):
                    print("Warning: NaN/Inf in cross dot product")
                    return self._get_zero_grads_A(), float('inf')

                g_cross_A = torch.autograd.grad(
                    dot_cross, A_params, retain_graph=False, allow_unused=True
                )
                g_cross_A = [torch.zeros_like(p) if g is None else g for p, g in zip(A_params, g_cross_A)]
                
                if any(torch.isnan(g).any() or torch.isinf(g).any() for g in g_cross_A):
                    print("Warning: NaN/Inf in second term gradient wrt A")
                    return self._get_zero_grads_A(), float('inf')
                    
            except Exception as e:
                print(f"Error computing second term: {e}")
                return self._get_zero_grads_A(), float('inf')

            # 5) 组合最终梯度
            grad_A = [gv - gc for gv, gc in zip(g_val_A, g_cross_A)]
            
            # 最终检查和裁剪
            max_grad_norm = 1.0
            for i, g in enumerate(grad_A):
                if torch.isnan(g).any() or torch.isinf(g).any():
                    print(f"Warning: NaN/Inf in final gradient for A parameter {i}")
                    grad_A[i] = torch.zeros_like(g)
                else:
                    # 梯度裁剪
                    grad_norm = torch.norm(g)
                    if grad_norm > max_grad_norm:
                        grad_A[i] = g * (max_grad_norm / grad_norm)

            L_val_item = L_val.item()
            return grad_A, L_val_item
            
        except Exception as e:
            print(f"Error in implicit meta grad computation: {e}")
            return self._get_zero_grads_A(), float('inf')

    def _get_zero_grads_A(self):
        """返回零梯度，用于错误情况"""
        A_params = list(self.A.parameters())
        return [torch.zeros_like(p) for p in A_params]

    def forward_step_with_params(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, params, model):
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
        model_args = [batch_x, batch_x_mark, dec_inp, batch_y_mark]
        if self.args.model in MODEL_REQUIRES_CYCLE:
            model_args.append(batch_cycle)
        model_args = tuple(model_args)
        if self.args.output_attention:
            outputs, attn = functional_call(model, params, model_args)
        else:
            outputs, attn = functional_call(model, params, model_args), None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return outputs, batch_y, attn

    def initialize_meta_tasks(self, train_data):
        self.meta_learning = False

        task_data_list = split_dataset_with_overlap(train_data, self.num_tasks, self.args.overlap_ratio)
        task_data_list = [split_dataset(task_data, r=0.7) for task_data in task_data_list]

        support_data_list = [td[0] for td in task_data_list]
        support_loader_list = [DataLoader(support_data, batch_size=self.args.auxi_batch_size, shuffle=True) for support_data in support_data_list]
        support_loader_list = [cycle(support_loader) for support_loader in support_loader_list]

        query_data_list = [td[1] for td in task_data_list]
        query_loader_list = [DataLoader(query_data, batch_size=self.args.auxi_batch_size, shuffle=True) for query_data in query_data_list]
        query_loader_list = [cycle(query_loader) for query_loader in query_loader_list]
        return support_loader_list, query_loader_list

    def meta_train(self, support_loader_list, query_loader_list, path, res_path):
        # 在meta train阶段，损失函数参数可训练，模型参数也需要梯度（用于inner loop）
        enable_grad(self.A)
        enable_grad(self.model)

        A_optim = self._select_optimizer(self.A, self.meta_lr, optim_type=self.args.meta_optim_type)
        A_scheduler = Scheduler(A_optim, self.args, self.args.warmup_steps)

        epoch_time = time.time()
        meta_step = 0
        for step in range(self.args.warmup_steps):
            meta_step = step + 1
            verbose = (meta_step % 100 == 0)
            task_val_losses = []

            meta_lr_cur = A_scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_lr', meta_lr_cur, meta_step)

            self.model.train()
            self.A.train()

            # 准备累积 A 的梯度
            A_optim.zero_grad()
            A_params = list(self.A.parameters())
            accum_grads_A = [torch.zeros_like(p) for p in A_params]

            # 遍历所有任务，累积meta loss
            for task_id, (support_loader, query_loader) in enumerate(zip(support_loader_list, query_loader_list)):
                task_model = self.task_models[task_id]

                # 检查模型参数是否包含nan
                for name, param in task_model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"Warning: NaN detected in model parameter {name}")
                        param.data.zero_()  # 重置为0

                # 初始化 fast params 与 init params（均为叶子张量，允许求导）
                init_params = {k: v.clone().detach().requires_grad_(True) for k, v in get_param_dict(task_model).items()}
                fast_params = {k: v.clone().detach().requires_grad_(True) for k, v in init_params.items()}

                # 采样一个支持集 batch 做 inner loop
                sup_batch = next(support_loader)
                fast_params = self._inner_adapt_with_prox(task_model, fast_params, init_params, sup_batch)

                # 采样一个查询集 batch
                qry_batch = next(query_loader)

                # iMAML 隐式梯度，返回对 A 的梯度和 L_val
                grad_A_task, L_val_item = self._implicit_meta_grad_A(
                    task_model, init_params, fast_params, sup_batch, qry_batch
                )
                # 检查梯度是否为nan
                for i, g in enumerate(grad_A_task):
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        print(f"Warning: NaN/Inf in gradient for A parameter {i}, task {task_id}")
                        grad_A_task[i] = torch.zeros_like(g)
                accum_grads_A = [ag + g for ag, g in zip(accum_grads_A, grad_A_task)]
                task_val_losses.append(L_val_item)

                self.writer.add_scalar(f'{self.pred_len}/meta_train/task_{task_id+1}_meta_loss', L_val_item, meta_step)
                if verbose:
                    print(f"\ttask: {task_id + 1}, total task: {self.num_tasks} | meta loss: {L_val_item:.7f}")

            # 统一进行损失函数参数的更新
            for p, g in zip(A_params, accum_grads_A):
                if p.grad is not None:
                    p.grad.zero_()
                p.grad = (g / max(1, self.num_tasks))

            A_optim.step()

            avg_val_loss = float(np.mean(task_val_losses))
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_loss', avg_val_loss, meta_step)
            log_heatmap(self.writer, get_projection(self.A), f'{self.pred_len}/cov_mat', meta_step)

            if verbose:
                print(f"Step: {meta_step} cost time: {time.time() - epoch_time:.2f}s")
                print(f"Step: {meta_step} | Avg Meta Loss: {avg_val_loss:.7f}")
                epoch_time = time.time()

            if self.args.lradj in ['TST']:
                A_scheduler.step(verbose=verbose)
            else:
                if verbose:
                    A_scheduler.step(avg_val_loss, meta_step // 100)

        best_A_path = os.path.join(path, 'A.pth')
        torch.save(self.A, best_A_path)
        projection = get_projection(self.A)
        plot_heatmap(projection, save_path=os.path.join(res_path, 'cov_matrix.pdf'))
        print(f'Statistics of learned covariance matrix: \033[92m(mean: {np.mean(projection):.4f}, std: {np.std(projection):.4f}, min: {np.min(projection):.4f}, max: {np.max(projection):.4f})\033[0m')

    def meta_test(self, train_loader, vali_data, vali_loader, criterion, path):
        if self.model_per_task and self.num_tasks > 1:
            del self.task_models[1:]

        disable_grad(self.A)
        enable_grad(self.model)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer(self.model, self.lr)
        scheduler = Scheduler(model_optim, self.args, train_steps)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss, train_loss_mse = [], []

            lr_cur = scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_test/lr', lr_cur, self.epoch)

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
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss', loss.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_mse', loss_mse.item(), self.step)

                if (i + 1) % 100 == 0:
                    print(f"\tMeta Test - iters: {i + 1}, epoch: {self.epoch} | loss: {loss.item():.7f}, mse loss: {loss_mse.item():.7f}")
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; cost time: {cost_time:.4f}s; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_mse = np.average(train_loss_mse)
            valid_loss_mse, valid_loss_cov = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_cov', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_mse', train_loss_mse, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_cov', valid_loss_cov, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_mse', valid_loss_mse, self.epoch)

            print(f"Epoch: {self.epoch} | Train Loss Cov: {train_loss:.7f}, MSE: {train_loss_mse:.7f} | Valid Loss Cov: {valid_loss_cov:.7f}, MSE: {valid_loss_mse:.7f}")
            early_stopping(valid_loss_mse, self.model, path)
            if early_stopping.early_stop:
                print("Meta Test Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(valid_loss_mse, self.epoch)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        support_loader_list, query_loader_list = self.initialize_meta_tasks(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        criterion = self._select_criterion()

        # ============ Meta Train 阶段：只训练损失函数 ============
        print("\n>>>>>>>Meta Training Phase\n")
        self.meta_train(support_loader_list, query_loader_list, path, res_path)
        print("\n>>>>>>>Meta Training Phase completed\n")

        # ============ ML3 Meta Test 阶段：重新初始化模型，使用学习到的损失函数训练 ============
        print("\n>>>>>>>Meta Test Phase\n")
        self.meta_test(train_loader, vali_data, vali_loader, criterion, path)
        print("\n>>>>>>>Meta Test Phase completed\n")

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
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        metrics = OrderedDict(zip(['mae', 'mse', 'rmse', 'mape', 'mspe', 'mre'], [mae, mse, rmse, mape, mspe, mre]))

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

        for k, v in full_metrics.items():
            self.writer.add_scalar(f'{self.pred_len}/test/{k}', v, self.epoch)
        self.writer.close()

        if self.output_log:
            log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
            payload = f"{setting}\n\n{line}\n\n"
            with open(log_path, mode="a", encoding="utf-8") as f:
                f.write(payload)

        # np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, cov_loss, rmse, mape, mspe, mre]))
        yaml.safe_dump(dict(full_metrics), open(os.path.join(res_path, 'metrics.yaml'), 'w'), default_flow_style=False, sort_keys=False)

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            yaml.dump(vars(self.args), open(os.path.join(res_path, 'config.yaml'), 'w'), default_flow_style=False)

        return
