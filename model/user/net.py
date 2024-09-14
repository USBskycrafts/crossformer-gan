import torch
import torch.nn as nn
from .crossformer import CrossformerEncoder, CrossformerDecoder
from .params import encoder_params, decoder_params
from typing import List
from tools.accuracy_tool import general_image_metrics


class UserNet(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super(UserNet, self).__init__()
        self.encoder = CrossformerEncoder(*encoder_params.values())
        self.decoder = CrossformerDecoder(*decoder_params.values())

        self.loss = nn.L1Loss()

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = torch.cat([data["t1"], data["t2"]], dim=1)
        features, shapes = self.encoder(x)
        pred = self.decoder(features, shapes)
        target = data["t1ce"]

        loss = self.loss(pred, target)
        acc_result = general_image_metrics(
            pred, target, config, acc_result)
        return {
            "loss": loss,
            "output": [acc_result],
            "acc_result": acc_result,
            "predict": pred
        }
