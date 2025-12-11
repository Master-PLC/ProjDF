import torch

from math import sqrt
import torch.nn.functional as F

def gaussian_kernel(X, Y, gamma=1.0):
    """
    高斯核函数 (Gaussian RBF Kernel)
    Formula: K(x, y) = exp(-||x-y||^2 * gamma)
    
    Args:
        X: Tensor [B, D_flat]
        Y: Tensor [M, D_flat] (M can be B or J)
        gamma: 核系数 (类似于 1/(2*sigma^2))
    Returns:
        K: Tensor [B, M]
    """
    if X.ndim > 2: X = X.reshape(X.size(0), -1)
    if Y.ndim > 2: Y = Y.reshape(Y.size(0), -1)
    dim = X.size(1)
        
    X_norm_sq = (X**2).sum(1).view(-1, 1)
    Y_norm_sq = (Y**2).sum(1).view(-1, 1)
    # 计算平方欧氏距离: ||x-y||^2 = x^2 + y^2 - 2xy
    squared_dist = X_norm_sq + Y_norm_sq.T - 2.0 * torch.mm(X, Y.T)
    
    # 防止数值下溢
    squared_dist = torch.clamp(squared_dist, min=0.0)
    return torch.exp(-squared_dist / dim * gamma)

def exponential_kernel(X, Y, gamma=1.0):
    """
    指数核函数 (Exponential / Laplacian Kernel)
    Formula: K(x, y) = exp(-||x-y|| * gamma)
    
    Args:
        X: Tensor [B, D_flat]
        Y: Tensor [M, D_flat]
    Returns:
        K: Tensor [B, M]
    """
    if X.ndim > 2: X = X.reshape(X.size(0), -1)
    if Y.ndim > 2: Y = Y.reshape(Y.size(0), -1)
    dim = X.size(1)
    
    X_norm_sq = (X**2).sum(1).view(-1, 1)
    Y_norm_sq = (Y**2).sum(1).view(-1, 1)
    squared_dist = X_norm_sq + Y_norm_sq.T - 2.0 * torch.mm(X, Y.T)
    
    # 限制最小值为 1e-8 防止 sqrt(0) 导致梯度 NaN
    dist = torch.sqrt(torch.clamp(squared_dist, min=1e-8))
    return torch.exp(-dist / sqrt(dim) * gamma)

# 核函数映射表
KERNEL_MAP = {
    'gau': gaussian_kernel,
    'exp': exponential_kernel
}

def akb_loss(pred, target, kernel_type='gau', gamma=0.1, J=3, inner_lr=0.05, inner_steps=3, optim_type='adam', solver_type='exact', reg=1e-3, C=0.01):
    """
    Adaptive Kernel Balancing (AKB) Loss Calculation Function.
    
    该函数执行以下步骤：
    1. 计算样本级的重构误差。
    2. 执行内层循环优化，寻找能解释这些误差的样本权重 alpha。
    3. 根据 alpha 选出 Top-J 个"困难样本"作为 Anchor。
    4. 计算预测值和真实值在这些 Anchor 上的核投影差异。
    """
    
    # 1. 检查并获取核函数
    if kernel_type not in KERNEL_MAP: raise ValueError(f"Unknown kernel: {kernel_type}")
    kernel_func = KERNEL_MAP[kernel_type]
    
    B = pred.size(0)
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)

    with torch.no_grad():
        # 1. 计算拟合目标
        # 注意：这里 pred 只是用来计算误差值，不需要传回梯度给模型
        # 如果不加 detach，虽然外面有 no_grad，但显式 detach 逻辑更清晰
        e_loss_per_sample = torch.mean((pred.detach() - target)**2, dim=(1, 2))
        
        # 2. 计算 Target 核矩阵
        K_yy = kernel_func(target_flat, target_flat, gamma)

        # 3. 计算 Alpha
        if solver_type == 'exact':
            # 闭式解：直接解线性方程组
            K_reg = K_yy + reg * torch.eye(B, device=pred.device)
            # unsqueeze/squeeze 是为了匹配矩阵乘法维度
            alpha = torch.linalg.solve(K_reg, e_loss_per_sample.unsqueeze(1)).squeeze()
        else:
            # 迭代解：需要局部梯度，但不需要模型梯度
            with torch.enable_grad():
                alpha = torch.full((B,), 1./B, device=pred.device, requires_grad=True)
                if optim_type == 'adam':
                    optimizer = torch.optim.Adam([alpha], lr=inner_lr)
                else:
                    optimizer = torch.optim.SGD([alpha], lr=inner_lr)
                
                for _ in range(inner_steps):
                    optimizer.zero_grad()
                    fitted_error = torch.mv(K_yy, alpha) # K_yy 是 constant
                    loss_fit = torch.mean((e_loss_per_sample - fitted_error)**2)
                    loss_fit.backward()
                    optimizer.step()

        # 4. 选择 Top-J 困难样本索引
        actual_J = min(J, B)
        _, topj_indices = torch.topk(torch.abs(alpha), k=actual_J)
        target_anchors = target_flat[topj_indices] # [J, D_flat]

    # 4. 计算投影差异
    pred_proj = kernel_func(pred_flat, target_anchors, gamma)     # [B, J]
    target_proj = kernel_func(target_flat, target_anchors, gamma) # [B, J]
    
    diff = torch.mean(pred_proj - target_proj, dim=0)
    loss = F.relu(diff - C) + F.relu(-diff - C)
    loss = loss.sum()
    return loss


def wkb_loss(pred, target, kernel_type='gau', gamma=0.1, J=3, inner_lr=0.05, inner_steps=3, optim_type='adam'):
    """
    Worst-case Kernel Balancing (WKB):
    通过内层循环寻找使得'预测分布'与'真实分布'差异最大的方向(对抗)。
    核心思想：Adversarial Training (对抗最坏情况)。
    对应原论文公式 (4.2) 及代码 MF_WKBIPS。
    """
    if kernel_type not in KERNEL_MAP: raise ValueError(f"Unknown kernel: {kernel_type}")
    kernel_func = KERNEL_MAP[kernel_type]
    
    B = pred.size(0)
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    
    # 1. 内层循环：寻找 alpha 以最大化分布差异
    # alpha 定义了 RKHS 中的最坏测试函数 f(x) = sum(epsilon_i * K(x, y_i))
    alpha = torch.zeros(B, device=pred.device, requires_grad=True, dtype=torch.float32)
    # 注意：这里可能需要更大的 lr 或者动量，因为是对抗过程
    if optim_type == 'adam':
        optimizer = torch.optim.Adam([alpha], lr=inner_lr)
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD([alpha], lr=inner_lr)
    else:
        raise ValueError(f"Unknown optimizer: {optim_type}")
    
    # 预计算核矩阵，用于内层循环加速
    with torch.no_grad():
        K_yy = kernel_func(target_flat, target_flat, gamma)
        # 注意：这里用 detach 的 pred 来计算矩阵，因为内层循环只优化 epsilon，不更新 pred 模型
        K_py = kernel_func(pred_flat.detach(), target_flat, gamma)
    
    with torch.enable_grad():
        for _ in range(inner_steps):
            optimizer.zero_grad()
            
            # 计算测试函数 f 在 Pred 和 Target 上的期望
            # E[f(Pred)] = Mean(K_py @ alpha)
            exp_f_pred = torch.mean(torch.mv(K_py, alpha))
            # E[f(Target)] = Mean(K_yy @ alpha)
            exp_f_target = torch.mean(torch.mv(K_yy, alpha))
            
            # 计算差异平方 (Numerator)
            worst_loss = (exp_f_pred - exp_f_target)**2
            
            # 计算 RKHS 范数平方 (Denominator) 用于归一化
            # Norm^2 = alpha^T * K_yy * alpha
            # norm_squared = torch.dot(alpha, torch.mv(K_yy, alpha))
            norm_squared = torch.sum(torch.mv(K_yy, alpha)**2)
            
            # 目标：最大化 (Diff^2 / Norm^2) => 最小化负值
            # 添加小量防止除零
            loss_adv = - worst_loss / (norm_squared + 1e-8)
            
            loss_adv.backward()
            optimizer.step()
            
    # 2. 选择 Anchors (Top-J)
    # 那些 alpha 权重大的点，就是造成分布差异最大的"罪魁祸首"
    with torch.no_grad():
        actual_J = min(J, B)
        _, topj_indices = torch.topk(torch.abs(alpha), k=actual_J)
        target_anchors = target_flat[topj_indices]
        
    # 3. 计算投影差异 (在对抗选出的最坏方向上)
    pred_proj = kernel_func(pred_flat, target_anchors, gamma)
    target_proj = kernel_func(target_flat, target_anchors, gamma)
    
    return pred_proj - target_proj


def rkb_loss(pred, target, kernel_type='gau', gamma=0.1, J=3, inner_lr=0.05, inner_steps=3, optim_type='adam'):
    """
    Random Kernel Balancing (RKB):
    随机选择 J 个样本作为 Anchors 计算差异。
    核心思想：基准 (Baseline)，假设随机采样能覆盖分布。
    """
    if kernel_type not in KERNEL_MAP: raise ValueError(f"Unknown kernel: {kernel_type}")
    kernel_func = KERNEL_MAP[kernel_type]
    
    B = pred.size(0)
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    
    # 1. 随机选择 Anchors
    actual_J = min(J, B)
    rand_indices = torch.randperm(B, device=pred.device)[:actual_J]
    target_anchors = target_flat[rand_indices]
    
    # 2. 计算投影差异
    pred_proj = kernel_func(pred_flat, target_anchors, gamma)
    target_proj = kernel_func(target_flat, target_anchors, gamma)
    
    return pred_proj - target_proj
