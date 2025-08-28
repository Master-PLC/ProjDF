import matplotlib.pyplot as plt
import ot
import torch
import torch_dct as dct
from ot.backend import get_backend
from ot.utils import dots, list_to_array


def cal_wasserstein(X1, X2, distance, ot_type='sinkhorn', normalize=1, mask_factor=0.01, numItermax=10000, stopThr=1e-4, reg_sk=0.1, reg_m=10):
    """
    X1: prediction sequence with shape [B, T, D]
    X2: label sequence with shape [B, T, D]
    distance: the method to calculate the pair-wise sample distance
    ot_type: the definition of OT problem
    reg_sk: the strength of entropy regularization in Sinkhorn
    reg_m: the strength of mass-preservation constraint in UOT
    Currently we can fix the ot_type as sinkhorn, and there are key parameters to tune:
        1. whether normalize is needed?
        2. which distance is better? time/fft_2norm/fft_1norm, other distances do not need to investigate now.
        3. the optimum reg_sk? Possible range: [0.01-5]
    Moreover, the weight of the loss should be tuned in this duration. Try both relative weights (\alpha for wass, 1-\alpha for mse) and absolute weights (\alpha for wass, 1 for mse)
    """
    B, T, D = X1.shape
    device = X1.device

    if distance == 'time':
        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
    elif distance == 'fft_2norm':
        X1 = torch.fft.rfft(X1.transpose(1, 2))
        X2 = torch.fft.rfft(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()**2).sum(-1)
    elif distance == 'fft_1norm':
        X1 = torch.fft.rfft(X1.transpose(1, 2))
        X2 = torch.fft.rfft(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()).sum(-1)
    elif distance == 'fft_2norm_2d':
        X1 = torch.fft.rfft2(X1.transpose(1, 2))  # [B, D, T//2+1]
        X2 = torch.fft.rfft2(X2.transpose(1, 2))  # [B, D, T//2+1]
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()**2).sum(-1)
    elif distance == 'fft_1norm_2d':
        X1 = torch.fft.rfft2(X1.transpose(1, 2))
        X2 = torch.fft.rfft2(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()).sum(-1)
    elif distance == 'dct_2norm':
        X1 = dct.dct(X1.transpose(1, 2), norm='ortho')  # [B, D, T]
        X2 = dct.dct(X2.transpose(1, 2), norm='ortho')  # [B, D, T]
        M = ((X1.flatten(1)[:, None, :] - X2.flatten(1)[None, :, :]) ** 2).sum(-1)
    elif distance == 'dct_1norm':
        X1 = dct.dct(X1.transpose(1, 2), norm='ortho')
        X2 = dct.dct(X2.transpose(1, 2), norm='ortho')
        M = (X1.flatten(1)[:, None, :] - X2.flatten(1)[None, :, :]).abs().sum(-1)
    elif distance == 'dct_2norm_2d':
        X1 = dct.dct_2d(X1.transpose(1, 2), norm='ortho')  # [B, D, T]
        X2 = dct.dct_2d(X2.transpose(1, 2), norm='ortho')  # [B, D, T]
        M = ((X1.flatten(1)[:, None, :] - X2.flatten(1)[None, :, :]) ** 2).sum(-1)
    elif distance == 'fft_mag':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
    elif distance == 'fft_mag_abs':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
        M = torch.norm(X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:], p=1, dim=2)
    elif distance == 'fft_2norm_kernel':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)
        sigma = 1
        M = -1 * torch.exp(-1 * ((X1[:,None,:] - X2[None,:,:]).abs()).sum(-1) / 2/sigma)
    elif distance == 'fft_2norm_multikernel':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)
        w = [0.5, 0.5]
        sigma = [0.1, 1]
        dist_weighted = [-1 * w[i] * torch.exp(-1 * ((X1[:,None,:] - X2[None,:,:]).abs()).sum(-1) / 2/sigma[i]) for i in range(len(w))]
        M = sum(dist_weighted)
    elif distance == 'fft_2norm_per_dim':
        X1 = torch.fft.rfft(X1.transpose(1, 2))  # [B, D, T//2+1]
        X2 = torch.fft.rfft(X2.transpose(1, 2))  # [B, D, T//2+1]
        
        # Correct calculation of M
        M = ((X1[:, None, :, :] - X2[None, :, :, :]).abs()**2).sum(-1)  # [B, B, D]
        
        if normalize == 1:
            # equals to M = torch.stack([M[..., d] / M[..., d].max() for d in range(D)], dim=-1)
            M = M / M.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]

        loss = 0
        a, b = torch.ones((B,), device=device) / B, torch.ones((B,), device=device) / B

        for d in range(D):
            M_d = M[:, :, d]

            if mask_factor > 0:
                mask = torch.ones_like(M_d) - torch.eye(B, device=device)
                M_d = M_d + mask_factor * mask

            if ot_type == 'sinkhorn':
                pi = ot.sinkhorn(a, b, M_d, reg=reg_sk, numItermax=numItermax, stopThr=stopThr)
            elif ot_type == 'emd':
                pi = ot.emd2(a, b, M_d, numItermax=numItermax)
            elif ot_type == 'uot':
                pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M_d, reg=reg_sk, stopThr=stopThr, numItermax=numItermax, reg_m=reg_m)
            elif ot_type == 'uot_mm':
                pi = ot.unbalanced.mm_unbalanced(a, b, M_d, reg_m=reg_m, numItermax=numItermax, stopThr=stopThr)

            loss += (pi * M_d).sum()

        loss = loss / D
        return loss

    elif distance == 'fft_wasserstein_1d':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)

        X1 = X1.abs()
        X2 = X2.abs()

        a, b = torch.ones((B, X1.shape[1]), device=device) / B, torch.ones((B, X2.shape[1]), device=device) / B
        loss = ot.lp.wasserstein_1d(X1, X2, a, b, p=2)
        loss = loss.mean()
        return loss

    elif distance == 'wasserstein_1d_per_dim':
        a, b = torch.ones((B, T), device=device) / B, torch.ones((B, T), device=device) / B

        loss = 0
        for d in range(D):
            X1_d = X1[..., d]
            X2_d = X2[..., d]
            loss += ot.lp.wasserstein_1d(X1_d, X2_d, a, b, p=2)
        loss = loss / D
        loss = loss.mean()
        return loss

    elif distance == 'fft_wasserstein_empirical':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)

        X1 = X1.abs()
        X2 = X2.abs()

        a, b = torch.ones((B,1), device=device, dtype=X1.dtype) / B, torch.ones((B,1), device=device, dtype=X2.dtype) / B
        loss = ot.gaussian.empirical_bures_wasserstein_distance(X1, X2, reg=reg_sk, ws=a, wt=b)
        loss = empirical_bures_wasserstein_distance(X1, X2, reg=reg_sk, ws=a, wt=b)
        loss = loss.mean()
        return loss

    elif distance == 'wasserstein_empirical_per_dim':
        #! 去掉rfft, 对each dimension考虑
        a, b = torch.ones((B,1), device=device, dtype=X1.dtype) / B, torch.ones((B,1), device=device, dtype=X2.dtype) / B

        loss = 0
        for d in range(D):
            X1_d = X1[..., d]
            X2_d = X2[..., d]
            # loss += ot.gaussian.empirical_bures_wasserstein_distance(X1_d, X2_d, reg=reg_sk, ws=a, wt=b)
            loss += empirical_bures_wasserstein_distance(X1_d, X2_d, reg=reg_sk, ws=a, wt=b)
        loss = loss / D
        loss = loss.mean()
        return loss

    if normalize == 1:
        M = M / M.max()

    if mask_factor > 0:
        mask = torch.ones_like(M) - torch.eye(B, device=device)
        M = M + mask_factor * mask

    a, b = torch.ones((B,), device=device) / B, torch.ones((B,), device=device) / B

    if ot_type == 'sinkhorn':
        pi = ot.sinkhorn(a, b, M, reg=reg_sk, max_iter=numItermax, tol_rel=stopThr).detach()
    elif ot_type == 'emd':
        pi = ot.emd(a, b, M, numItermax=numItermax).detach()
    elif ot_type == 'uot':
        pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg_sk, stopThr=stopThr, numItermax=numItermax, reg_m=reg_m).detach()
    elif ot_type == 'uot_mm':
        pi = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, c=None, reg=0, div='kl', G0=None, numItermax=numItermax, stopThr=stopThr).detach()

    loss = (pi * M).sum()
    return loss


def empirical_bures_wasserstein_distance(
    xs, xt, reg=1e-6, ws=None, wt=None, bias=True, log=False
):
    r"""copy from pot library"""
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    d = xs.shape[1]

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    if bias:
        mxs = nx.dot(ws.T, xs) / nx.sum(ws)
        mxt = nx.dot(wt.T, xt) / nx.sum(wt)

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = nx.zeros((1, d), type_as=xs)
        mxt = nx.zeros((1, d), type_as=xs)

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws) + reg * nx.eye(d, type_as=xs)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt) + reg * nx.eye(d, type_as=xt)

    if log:
        W, log = bures_wasserstein_distance(mxs, mxt, Cs, Ct, log=log, eps=reg)
        log["Cs"] = Cs
        log["Ct"] = Ct
        return W, log
    else:
        W = bures_wasserstein_distance(mxs, mxt, Cs, Ct, eps=reg)
        return W


def bures_wasserstein_distance(ms, mt, Cs, Ct, log=False, eps=1e-6, method='svd_stable'):
    r"""copy from pot library"""
    ms, mt, Cs, Ct = list_to_array(ms, mt, Cs, Ct)
    nx = get_backend(ms, mt, Cs, Ct)

    def cal_distance(ms, mt, Cs, Ct, nx, sqrtm_func, log=False, eps=1e-6):
        Cs12 = sqrtm_func(Cs, eps=eps)
        B = nx.trace(Cs + Ct - 2 * sqrtm_func(dots(Cs12, Ct, Cs12), eps=eps))
        W = nx.sqrt(nx.maximum(nx.norm(ms - mt) ** 2 + B, 0))
        if torch.isnan(W) or torch.isinf(W):
            raise ValueError("nan/inf distance")

        if log:
            log = {}
            log["Cs12"] = Cs12
            return W, log
        else:
            return W

    try:
        return cal_distance(ms, mt, Cs, Ct, nx, sqrtm_svd_stable, log=log, eps=eps)
    except Exception:
        try:
            return cal_distance(ms, mt, Cs, Ct, nx, sqrtm, log=log, eps=eps)
        except Exception:
            return cal_distance(ms, mt, Cs, Ct, nx, sqrtm_newton_schulz_stable, log=log, eps=eps)


def sqrtm(a, *args, **kwargs):
    L, V = torch.linalg.eigh(a)
    L = torch.sqrt(L)
    # Q[...] = V[...] @ diag(L[...])
    Q = torch.einsum("...jk,...k->...jk", V, L)
    # R[...] = Q[...] @ V[...].T
    return torch.einsum("...jk,...kl->...jl", Q, torch.transpose(V, -1, -2))


def sqrtm_svd(a, eps=1e-8, *args, **kwargs):
    U, S, Vh = torch.linalg.svd(a)
    S = torch.clamp(S, min=eps)
    S_root = torch.sqrt(S)
    return (U * S_root) @ U.t()


def sqrtm_svd_stable(A: torch.Tensor, eps: float = 1e-8, dtype: torch.dtype = torch.float64, *args, **kwargs):
    """稳定SVD法（对称化+双精度+奇异值截断）"""
    A = (A + A.transpose(-1, -2)) * 0.5  # 对称化
    A64 = A.to(dtype)  # 转double提升稳定性
    U, S, Vh = torch.linalg.svd(A64, full_matrices=False)
    S = torch.clamp(S, min=eps)
    S_root = torch.sqrt(S)
    sqrtA64 = (U * S_root[..., None, :]) @ U.transpose(-1, -2)
    return sqrtA64.to(A.dtype)  # 回转原始dtype


def sqrtm_newton_schulz(A: torch.Tensor, num_iters: int = 20, eps: float = 1e-12, *args, **kwargs):
    """牛顿-舒尔茨迭代法"""
    A = (A + A.transpose(-1, -2)) * 0.5 + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)  # 对称化+正则化
    norm = torch.norm(A, p='fro')
    Y = A / norm  # 归一化（谱范数≈1）
    Z = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    
    for _ in range(num_iters):
        T = 0.5 * (3.0 * Z - Y @ Z @ Y)
        Z = T
        Y = 0.5 * (3.0 * Y - Y @ Y @ Y)

    return torch.sqrt(norm) * Y  # 恢复尺度


def sqrtm_newton_schulz_stable(A: torch.Tensor, num_iters: int = 20, eps: float = 1e-12, *args, **kwargs) -> torch.Tensor:
    """
    改进的Newton‑Schulz迭代：先缩放矩阵使其谱范数≈1，并添加正则化
    """
    # 1. 对称化（消除非对称性）
    A = (A + A.transpose(-1, -2)) * 0.5
    d = A.shape[-1]
    
    # 2. 正定化：添加微小正则项（避免奇异矩阵）
    jitter = eps * torch.eye(d, dtype=A.dtype, device=A.device)
    A = A + jitter
    
    # 3. 缩放：使谱范数接近1（通过Frobenius范数缩放）
    norm = torch.norm(A, p='fro')
    if norm > 0:
        A = A / norm  # 现在A的谱范数≤1（正定矩阵的谱范数≤Frobenius范数）
    else:
        return torch.eye(d, dtype=A.dtype, device=A.device)
    
    # 4. 迭代计算（Newton-Schulz）
    Y = A
    Z = torch.eye(d, dtype=A.dtype, device=A.device)
    for _ in range(num_iters):
        T = 0.5 * (3.0 * Z - Y @ Z @ Y)
        Z = T
        Y = 0.5 * (3.0 * Y - Y @ Y @ Y)
    
    # 5. 恢复尺度（sqrt(A) = sqrt(norm) * Y）
    sqrtA = torch.sqrt(norm) * Y
    return sqrtA