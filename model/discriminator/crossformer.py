import torch
import torch.nn as nn
from model.user.crossformer import CrossformerEncoder
from model.user.params import encoder_params
from copy import deepcopy


class Discriminator(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super(Discriminator, self).__init__()
        cross_params = encoder_params["cross_params"]
        crossformer_params = encoder_params["transformer_params"]
        cross_params = deepcopy(cross_params)
        cross_params[0].input_dim = 1
        self.encoder = CrossformerEncoder(
            cross_params=cross_params, crossformer_params=crossformer_params)

        self.classify_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.bce_loss = nn.BCELoss()

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        fake, real = data["fake"], data["real"]
        f_features, f_shapes = self.encoder(fake)
        r_features, r_shapes = self.encoder(real)
        pred_fake = self.classify_head(f_features[-1])
        pred_real = self.classify_head(r_features[-1])

        loss = self.bce_loss(pred_fake, torch.zeros_like(pred_fake)) + \
            self.bce_loss(pred_real, torch.ones_like(pred_real))

        return {
            "loss": loss,
            "output": [acc_result],
            "acc_result": acc_result,
            "pred": {
                "fake": pred_fake,
                "real": pred_real
            }
        }
