import os
import time
import torch
import warnings
import yaml

import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNorm

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, Scheduler

warnings.filterwarnings('ignore')


def add_sn(layer, specral_norm=False):
    return SpectralNorm(layer) if specral_norm else layer


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ind_discr = args.ind_discr
        spectral_norm = args.spectral_norm
        window = args.seq_len + args.pred_len
        self.head = nn.Sequential(
            add_sn(nn.Linear(window, 512), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.seq_processor = nn.Sequential(
            add_sn(nn.Linear(512, 128), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            add_sn(nn.Linear(128, 1), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.var_processor = add_sn(nn.Linear(args.c_out, 1), spectral_norm)

    def forward(self, z):
        z = z.transpose(1, 2)
        z = self.head(z)
        z = self.seq_processor(z).squeeze()
        score = self.var_processor(z)
        return score


class Exp_Long_Term_Forecast_GAN_OneStep(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.discriminator = Discriminator(args).to(self.device)
        self.view_lambda = args.view_lambda
        self.label_smoothing = args.label_smoothing
        self.adversarial_learning = False
        assert args.auxi_lambda
        # if not self.use_amp:
        #     self.model = torch.compile(self.model)
        #     self.discriminator = torch.compile(self.discriminator)

    def check_adversarial_learning(self):
        if self.args.fixed_step and self.step > self.args.fixed_step and not self.adversarial_learning:
            print(f"\n>>>>>>>Adversarial learning enabled at step {self.step}, epoch {self.epoch}\n")
            self.adversarial_learning = True

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
        discr_state_last_effective = None
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer_G = self._select_optimizer()
        scheduler_G = Scheduler(optimizer_G, self.args, train_steps)
        criterion_G = self._select_criterion()
        if self.use_amp:
            scaler_G = GradScaler()

        optimizer_D = self._select_optimizer(self.discriminator, lr=self.args.meta_lr, optim_type=self.args.meta_optim_type)
        scheduler_D = Scheduler(optimizer_D, self.args, train_steps, lradj=self.args.meta_lradj)
        criterion_D = self._select_criterion(self.args.meta_loss)
        if self.use_amp:
            scaler_D = GradScaler()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0

            train_loss_G, train_loss_D = [], []
            rec_losses, auxi_losses = [], []
            real_losses, fake_losses = [], []

            lr_cur_G = scheduler_G.get_lr()
            lr_cur_G = lr_cur_G[0] if isinstance(lr_cur_G, list) else lr_cur_G
            lr_cur_D = scheduler_D.get_lr()
            lr_cur_D = lr_cur_D[0] if isinstance(lr_cur_D, list) else lr_cur_D
            self.writer.add_scalar(f'{self.pred_len}/train_G/lr', lr_cur_G, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train_D/lr', lr_cur_D, self.epoch)

            self.model.train()
            self.discriminator.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1

                if 0 < self.label_smoothing < 1:
                    real = torch.ones(self.args.batch_size, 1, device=self.device) * (1 - self.label_smoothing)
                    fake = torch.ones(self.args.batch_size, 1, device=self.device) * self.label_smoothing
                else:
                    real = torch.ones(self.args.batch_size, 1, device=self.device)
                    fake = torch.zeros(self.args.batch_size, 1, device=self.device)

                self.check_adversarial_learning()

                #-------------------------------------------------------------------
                # Train the generator 
                #-------------------------------------------------------------------
                loss = 0
                if self.use_amp:
                    with autocast():
                        outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                        loss_rec = criterion_G(outputs, batch_y)
                        loss += self.args.rec_lambda * loss_rec if self.adversarial_learning else loss_rec

                    if self.adversarial_learning:
                        outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1).float()  # [B, S+P, D]
                        batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1).float()  # [B, S+P, D]

                        with autocast():
                            score = self.discriminator(outputs)
                            loss_auxi = criterion_D(score, real)
                            loss += self.args.auxi_lambda * loss_auxi
                    else:
                        loss_auxi = torch.tensor(1e4, device=self.device)

                    optimizer_G.zero_grad()
                    scaler_G.scale(loss).backward()
                    scaler_G.step(optimizer_G)
                    scaler_G.update()
                else:
                    outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                    loss_rec = criterion_G(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec if self.adversarial_learning else loss_rec

                    if self.adversarial_learning:
                        outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1).float()  # [B, S+P, D]
                        batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1).float()  # [B, S+P, D]

                        score = self.discriminator(outputs)
                        loss_auxi = criterion_D(score, real)
                        loss += self.args.auxi_lambda * loss_auxi
                    else:
                        loss_auxi = torch.tensor(1e4, device=self.device)

                    optimizer_G.zero_grad()
                    loss.backward()
                    optimizer_G.step()

                rec_losses.append(loss_rec.item())
                auxi_losses.append(loss_auxi.item())
                train_loss_G.append(loss.item())

                self.writer.add_scalar(f'{self.pred_len}/train_G_iter/loss_rec', loss_rec.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_G_iter/loss_auxi', loss_auxi.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_G_iter/loss', loss.item(), self.step)

                #-------------------------------------------------------------------
                # Train the discriminator
                #-------------------------------------------------------------------
                loss = 0
                if self.adversarial_learning:
                    if self.use_amp:
                        with autocast():
                            real_score = self.discriminator(batch_y)
                            real_loss = criterion_D(real_score, real)

                            fake_score = self.discriminator(outputs.detach())
                            fake_loss = criterion_D(fake_score, fake)
                            loss += 0.5 * real_loss + 0.5 * fake_loss

                        optimizer_D.zero_grad()
                        scaler_D.scale(loss).backward()
                        scaler_D.step(optimizer_D)
                        scaler_D.update()
                    else:
                        real_score = self.discriminator(batch_y)
                        real_loss = criterion_D(real_score, real)

                        fake_score = self.discriminator(outputs.detach())
                        fake_loss = criterion_D(fake_score, fake)
                        loss += 0.5 * real_loss + 0.5 * fake_loss

                        optimizer_D.zero_grad()
                        loss.backward()
                        optimizer_D.step()
                else:
                    real_loss = torch.tensor(1e4, device=self.device)
                    fake_loss = torch.tensor(1e4, device=self.device)
                    loss = torch.tensor(1e4, device=self.device)

                real_losses.append(real_loss.item())
                fake_losses.append(fake_loss.item())
                train_loss_D.append(loss.item())

                self.writer.add_scalar(f'{self.pred_len}/train_D_iter/loss_real', real_loss.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_D_iter/loss_fake', fake_loss.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_D_iter/loss', loss.item(), self.step)

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {}, epoch: {} | loss_rec: {:.7f}, loss_auxi: {:.7f}, loss_G: {:.7f} | loss_real: {:.7f}, loss_fake: {:.7f}, loss_D: {:.7f}".format(
                            i + 1, self.epoch, loss_rec.item(), loss_auxi.item(), train_loss_G[-1], real_loss.item(), fake_loss.item(), train_loss_D[-1]
                        )
                    )
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj in ['TST']:
                    scheduler_G.step(verbose=(i + 1 == train_steps))
                    scheduler_D.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss_G = np.average(train_loss_G); train_loss_D = np.average(train_loss_D)
            rec_loss = np.average(rec_losses); auxi_loss = np.average(auxi_losses)
            real_loss = np.average(real_losses); fake_loss = np.average(fake_losses)
            vali_loss = self.vali(vali_data, vali_loader, criterion_G)

            self.writer.add_scalar(f'{self.pred_len}/train_G/loss', train_loss_G, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train_G/loss_rec', rec_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train_G/loss_auxi', auxi_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train_D/loss', train_loss_D, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train_D/loss_real', real_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train_D/loss_fake', fake_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)

            print(
                "Epoch: {}, Steps: {} | TrainG Loss: {:.7f} Loss_rec: {:.7f} Loss_auxi: {:.7f} | TrainD Loss: {:.7f} Loss_real: {:.7f} Loss_fake: {:.7f} | Vali Loss: {:.7f}"
                .format(self.epoch, self.step, train_loss_G, rec_loss, auxi_loss, train_loss_D, real_loss, fake_loss, vali_loss)
            )
            other_to_save = {'discriminator': self.discriminator}
            improved = early_stopping(vali_loss, self.model, path, **other_to_save)
            self.args.learned_from_method = True if improved and self.adversarial_learning else False

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler_G.step(vali_loss, self.epoch)
                scheduler_D.step(vali_loss, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.discriminator = torch.load(os.path.join(path, 'discriminator.pth'))

        return self.model
