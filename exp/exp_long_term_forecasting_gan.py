import os
import time
import warnings

import numpy as np
import yaml
from copy import deepcopy
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import MODEL_REQUIRES_CYCLE
from torch import optim
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.tools import EarlyStopping, visual, Scheduler, adjust_learning_rate

warnings.filterwarnings('ignore')


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        window = args.seq_len + args.pred_len
        self.tmp_head = nn.Sequential(
            nn.Linear(window, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.feq_head = nn.Sequential(
            nn.Linear(window // 2 + 1, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.ind_discr = args.ind_discr
        if self.ind_discr:
            self.seq_processor = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 128), nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(128, 1), nn.LeakyReLU(0.2, inplace=True),
                ) for _ in range(2)
            ])
            self.var_processor = nn.ModuleList([nn.Linear(args.c_out, 1) for _ in range(2)])
        else:
            self.seq_processor = nn.Sequential(
                nn.Linear(512, 128), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 1), nn.LeakyReLU(0.2, inplace=True),
            )
            self.var_processor = nn.Linear(args.c_out, 1)

    def forward(self, z):
        z = z.transpose(1, 2)
        tmp_z = self.tmp_head(z)
        feq_z = torch.fft.rfft(z, dim=-1).abs()
        feq_z = self.feq_head(feq_z)

        if self.ind_discr:
            tmp_z = self.seq_processor[0](tmp_z).squeeze()
            feq_z = self.seq_processor[1](feq_z).squeeze()
            tmp_score = self.var_processor[0](tmp_z)
            feq_score = self.var_processor[1](feq_z)
        else:
            tmp_z = self.seq_processor(tmp_z).squeeze()
            feq_z = self.seq_processor(feq_z).squeeze()
            tmp_score = self.var_processor(tmp_z)
            feq_score = self.var_processor(feq_z)
        return tmp_score, feq_score


class Exp_Long_Term_Forecast_GAN(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.discriminator = Discriminator(args).to(self.device)
        self.view_lambda = args.view_lambda
        assert args.auxi_lambda
        self.use_amp = args.use_amp
        # if not self.use_amp:
        #     self.model = torch.compile(self.model)
        #     self.discriminator = torch.compile(self.discriminator)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                if self.use_amp:
                    with autocast():
                        outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                else:
                    outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        # total_loss = np.average(total_loss)
        total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
        self.model.train()
        return total_loss

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
        scheduler_D = Scheduler(optimizer_D, self.args, train_steps)
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
            self.writer.add_scalar(f'{self.pred_len}/train_G/lr', lr_cur_G, self.epoch)

            lr_cur_D = scheduler_D.get_lr()
            lr_cur_D = lr_cur_D[0] if isinstance(lr_cur_D, list) else lr_cur_D
            self.writer.add_scalar(f'{self.pred_len}/train_D/lr', lr_cur_D, self.epoch)

            self.model.train()
            self.discriminator.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1

                real = torch.ones(self.args.batch_size, 1, device=self.device)
                fake = torch.zeros(self.args.batch_size, 1, device=self.device)

                #-------------------------------------------------------------------
                # Train the generator 
                #-------------------------------------------------------------------
                loss = 0
                if self.use_amp:
                    with autocast():
                        outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                        loss_rec = criterion_G(outputs, batch_y)
                        loss += self.args.rec_lambda * loss_rec

                    outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1).float()  # [B, S+P, D]
                    batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1).float()  # [B, S+P, D]

                    with autocast():
                        tmp_score, feq_score = self.discriminator(outputs)
                        tmp_loss = criterion_D(tmp_score, real)
                        feq_loss = criterion_D(feq_score, real)
                        loss_auxi = self.view_lambda * tmp_loss + (1 - self.view_lambda) * feq_loss
                        loss += self.args.auxi_lambda * loss_auxi

                    optimizer_G.zero_grad()
                    scaler_G.scale(loss).backward()
                    scaler_G.step(optimizer_G)
                    scaler_G.update()
                else:
                    outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                    loss_rec = criterion_G(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec

                    outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1).float()  # [B, S+P, D]
                    batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1).float()  # [B, S+P, D]

                    tmp_score, feq_score = self.discriminator(outputs)
                    tmp_loss = criterion_D(tmp_score, real)
                    feq_loss = criterion_D(feq_score, real)
                    loss_auxi = self.view_lambda * tmp_loss + (1 - self.view_lambda) * feq_loss
                    loss += self.args.auxi_lambda * loss_auxi

                    optimizer_G.zero_grad()
                    loss.backward()
                    optimizer_G.step()

                rec_losses.append(loss_rec.item())
                auxi_losses.append(loss_auxi.item())
                train_loss_G.append(loss.item())

                # self.writer.add_scalar(f'{self.pred_len}/train_G/loss_rec_iter', loss_rec, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_G/loss_tmp_iter', tmp_loss, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_G/loss_feq_iter', feq_loss, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_G/loss_auxi_iter', loss_auxi, self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_G/loss_iter', loss.item(), self.step)

                #-------------------------------------------------------------------
                # Train the discriminator
                #-------------------------------------------------------------------
                loss = 0
                if self.use_amp:
                    with autocast():
                        real_tmp_score, real_feq_score = self.discriminator(batch_y)
                        real_tmp_loss = criterion_D(real_tmp_score, real)
                        real_feq_loss = criterion_D(real_feq_score, real)
                        real_loss = self.view_lambda * real_tmp_loss + (1 - self.view_lambda) * real_feq_loss
                        loss += 0.5 * real_loss

                        fake_tmp_score, fake_feq_score = self.discriminator(outputs.detach())
                        fake_tmp_loss = criterion_D(fake_tmp_score, fake)
                        fake_feq_loss = criterion_D(fake_feq_score, fake)
                        fake_loss = self.view_lambda * fake_tmp_loss + (1 - self.view_lambda) * fake_feq_loss
                        loss += 0.5 * fake_loss

                    optimizer_D.zero_grad()
                    scaler_D.scale(loss).backward()
                    scaler_D.step(optimizer_D)
                    scaler_D.update()
                else:
                    real_tmp_score, real_feq_score = self.discriminator(batch_y)
                    real_tmp_loss = criterion_D(real_tmp_score, real)
                    real_feq_loss = criterion_D(real_feq_score, real)
                    real_loss = self.view_lambda * real_tmp_loss + (1 - self.view_lambda) * real_feq_loss
                    loss += 0.5 * real_loss

                    fake_tmp_score, fake_feq_score = self.discriminator(outputs.detach())
                    fake_tmp_loss = criterion_D(fake_tmp_score, fake)
                    fake_feq_loss = criterion_D(fake_feq_score, fake)
                    fake_loss = self.view_lambda * fake_tmp_loss + (1 - self.view_lambda) * fake_feq_loss
                    loss += 0.5 * fake_loss

                    optimizer_D.zero_grad()
                    loss.backward()
                    optimizer_D.step()

                real_losses.append(real_loss.item())
                fake_losses.append(fake_loss.item())
                train_loss_D.append(loss.item())

                # self.writer.add_scalar(f'{self.pred_len}/train_D/loss_real_tmp_iter', real_tmp_loss, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_D/loss_real_feq_iter', real_feq_loss, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_D/loss_real_iter', real_loss, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_D/loss_fake_tmp_iter', fake_tmp_loss, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_D/loss_fake_feq_iter', fake_feq_loss, self.step)
                # self.writer.add_scalar(f'{self.pred_len}/train_D/loss_fake_iter', fake_loss, self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_D/loss_iter', loss, self.step)

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
            early_stopping(vali_loss, self.model, path, **other_to_save)
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
            # self.discriminator = torch.load(os.path.join(ckpt_dir, 'discriminator.pth'))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                if self.use_amp:
                    with autocast():
                        outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                else:
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
            if self.args.auxi_mode == 'basis' and self.args.auxi_type == 'pca':
                train_data, _ = self._get_data(flag='train')
                pca_components = train_data.pca_components
                np.save(os.path.join(res_path, 'pca_components.npy'), pca_components)

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return
