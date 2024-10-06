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

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data):
        features, shapes = self.encoder(data)
        pred = self.decoder(features, shapes)

        return pred
