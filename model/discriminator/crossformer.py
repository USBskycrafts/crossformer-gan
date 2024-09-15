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

    def init_multi_gpu(self, device, config, *args, **kwargs):
        pass

    def forward(self, data):
        f_features, f_shapes = self.encoder(data)
        return f_features
