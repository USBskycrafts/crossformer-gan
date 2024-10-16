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
        self.header = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(2, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7),
            nn.Softplus()
        )

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data):
        data = self.header(data)
        features, shapes = self.encoder(data)
        pred = self.decoder(features, shapes)

        return self.tail(pred)
