# -*- coding: utf-8 -*-
"""
DBLoss: Decomposition-Based Loss (trend + season dual loss)
Adapted from: https://github.com/... (DBLoss repository)
    ts_benchmark/baselines/utils.py
"""
import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series.
    Input:  x of shape [B, T, D]
    Output: EMA(x) of shape [B, T, D]
    """

    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        _, t, _ = x.shape
        device = x.device
        powers = torch.flip(torch.arange(t, dtype=torch.double, device=device), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)


class DECOMP(nn.Module):
    """
    Series decomposition block: returns (season, trend).
    """

    def __init__(self, alpha):
        super(DECOMP, self).__init__()
        self.ma = EMA(alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class DBLoss(nn.Module):
    """
    Decomposition-Based dual loss (trend + season).
        - alpha: EMA smoothing coefficient used to decompose the series
        - beta : weight between season loss (MSE) and trend loss (MAE)

    L = beta * MSE(season) + (1 - beta) * scaled_MAE(trend)
    where scaled_MAE(trend) is rescaled by season_loss / trend_loss (stop-grad)
    so that the two terms lie on the same order of magnitude.
    """

    def __init__(self, alpha, beta):
        super().__init__()
        self.decomp = DECOMP(alpha)
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")
        self.mae = nn.L1Loss(reduction="mean")

    def forward(self, pred, target):
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        season_loss = self.mse(pred_season, target_season)
        trend_loss = self.mae(pred_trend, target_trend)
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
        return self.beta * season_loss + (1 - self.beta) * trend_loss
