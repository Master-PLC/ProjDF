import os
import time
import torch
import warnings
import yaml

from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
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
        pred_len = self.pred_len
        
        # =======================================================
        # 1. 定义 Chunk 执行函数
        # =======================================================
        # 这个函数会在显存中跑 chunk_size 步，这几步之间保留梯度图（快），
        # 但块与块之间通过 Checkpoint 连接（省显存）
        def run_chunk(start_step, chunk_steps, x_input, marks_tensor, batch_cycle_tensor=None):
            chunk_preds = []
            temp_x = x_input
            
            for k in range(chunk_steps):
                abs_step = start_step + k
                
                # 准备输入
                dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]], device=self.device)
                dec_inp = torch.cat([temp_x[:, -label_len:, :], dec_inp], dim=1)

                if marks_tensor is not None:
                    cur_x_mark = marks_tensor[:, abs_step : abs_step + seq_len, :]
                    enc_end = abs_step + seq_len
                    cur_y_mark = marks_tensor[:, enc_end - label_len : enc_end + 1, :]
                else:
                    cur_x_mark, cur_y_mark = None, None

                # 模型前向
                model_args = [temp_x, cur_x_mark, dec_inp, cur_y_mark]
                if batch_cycle_tensor is not None:
                    model_args.append(batch_cycle_tensor)

                if self.args.output_attention:
                    out, _ = self.model(*model_args)
                else:
                    out = self.model(*model_args)
                
                step_pred = out[:, -1:, :]
                chunk_preds.append(step_pred)
                
                # 更新 temp_x
                temp_x = torch.cat([temp_x, step_pred], dim=1)
                if temp_x.shape[1] > seq_len:
                    temp_x = temp_x[:, -seq_len:, :]
            
            # 返回这一块的所有预测结果，以及最后一个状态的 x (用于传递给下一个块)
            # 必须把 cat 后的 Tensor 返回，才能保持梯度链
            return torch.cat(chunk_preds, dim=1), temp_x

        # =======================================================
        # 2. 训练模式：分块 Checkpoint
        # =======================================================
        if self.model.training:
            # 建议 chunk_size 设置为 12 或 16
            # 如果显存还够，可以设大一点 (比如 24)，越大概率越快
            if 'ECL' in self.args.data_id:
                chunk_size = 24
            elif 'Traffic' in self.args.data_id:
                chunk_size = 12
            else:
                chunk_size = 96
            
            total_preds = []
            
            # 确保第一个输入有梯度，满足 checkpoint 要求
            if not curr_x.requires_grad:
                curr_x.requires_grad_(True)

            steps_remaining = pred_len
            current_step = 0

            while steps_remaining > 0:
                this_chunk_size = min(chunk_size, steps_remaining)
                
                # 准备 Checkpoint 参数
                args_tuple = (current_step, this_chunk_size, curr_x, full_marks)
                if self.args.model in MODEL_REQUIRES_CYCLE:
                    args_tuple += (batch_cycle,)
                else:
                    args_tuple += (None,)

                # 执行 Checkpoint
                # 这里的 run_chunk 内部是正常的反向传播图，速度快
                # Checkpoint 只发生在块与块之间
                chunk_out, next_x = checkpoint(run_chunk, *args_tuple, use_reentrant=False)
                
                total_preds.append(chunk_out)
                curr_x = next_x # 传递给下一轮
                
                current_step += this_chunk_size
                steps_remaining -= this_chunk_size
            
            outputs = torch.cat(total_preds, dim=1)
            attn = None # Checkpoint 模式下无法拿到 attn

        # =======================================================
        # 3. 推理模式：直接循环 (为了拿到 attn 和更少的 overhead)
        # =======================================================
        else:
            preds = []
            for i in range(pred_len):
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

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
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
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                loss = 0
                if self.args.rec_lambda:
                    loss_rec = criterion(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)
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
                self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi, self.step)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Loss is NaN or Inf, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                train_loss.append(loss.item())
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
