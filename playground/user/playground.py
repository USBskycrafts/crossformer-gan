from torch.nn import L1Loss, MSELoss
from playground.playground import Playground
import torch
import numpy as np
from model.user.net import UserNet
from tools.accuracy_tool import general_image_metrics
from tools.printer import Printer
from model.loss import GramLoss, ReconstructionLoss


class UserPlayground(Playground):
    def __init__(self, config,
                 models,
                 optimizers,
                 writer):
        super().__init__(config, models, optimizers, writer)

        self.generator = models['UserNet']
        # self.discriminator = models['discriminator']
        self.l1_loss = L1Loss()
        self.printer = Printer(config)

    def train(self, data, config, gpu_list, acc_result, mode):
        pred = self.generator(torch.cat([data['t1'], data['t2']], dim=1))
        target = data['t1ce']
        # fake_features = self.discriminator(pred)
        g_loss = self.l1_loss(
            pred, target)
        # g_loss += self.mse_loss(fake_features[-1],
        #                         torch.ones_like(fake_features[-1]))
        acc_result = general_image_metrics(pred, target, config, acc_result)
        yield {
            "name": "UserNet",
            "loss": g_loss,
            "acc_result": acc_result
        }
        # del fake_features
        # fake_features = self.discriminator(pred.detach())
        # target_features = self.discriminator(target)
        # d_loss = 0.5 * (self.mse_loss(fake_features[-1],
        #                               torch.zeros_like(fake_features[-1]))
        #                 + self.mse_loss(target_features[-1],
        #                                 torch.ones_like(target_features[-1]))) * 5
        # yield {
        #     "name": "discriminator",
        #     "loss": d_loss,
        #     "acc_result": acc_result
        # }
        self.writer.add_scalar("train/UserNet", g_loss, self.train_step)
        # self.writer.add_scalar("train/discriminator", d_loss, self.train_step)
        self.writer.add_scalar(
            "train/PSNR", np.mean(acc_result["PSNR"]), self.train_step)
        self.writer.add_scalar(
            "train/SSIM", np.mean(acc_result["SSIM"]), self.train_step)

    def test(self, data, config, gpu_list, acc_result, mode):
        pred = self.generator(torch.cat([data['t1'], data['t2']], dim=1))
        target = data['t1ce']
        acc_result = general_image_metrics(pred, target, config, acc_result)
        self.printer({
            **data,
            "pred": pred,
        }, self.test_step)
        return {
            "output": [acc_result],
            "acc_result": acc_result
        }

    def eval(self, data, config, gpu_list, acc_result, mode):
        pred = self.generator(torch.cat([data['t1'], data['t2']], dim=1))
        target = data['t1ce']
        loss = self.l1_loss(pred, target)
        acc_result = general_image_metrics(pred, target, config, acc_result)
        self.writer.add_scalar("eval/UserNet", loss, self.eval_step)
        self.writer.add_scalar(
            "eval/PSNR", np.mean(acc_result["PSNR"]), self.eval_step)
        self.writer.add_scalar(
            "eval/SSIM", np.mean(acc_result["SSIM"]), self.eval_step)
        if self.eval_step % 10 == 0:
            self.printer({
                **data,
                "pred": pred,
            }, self.eval_step)
        return {
            "loss": loss,
            "acc_result": acc_result
        }
