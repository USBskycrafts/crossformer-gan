import torch
import torch.nn as nn
from .crossformer_layer import CrossformerPack
from .cross_embedding import CrossScaleEmbedding
from typing import Dict, Any, List, Tuple


class CrossScaleParams:
    def __init__(self, input_dim: int,
                 output_dim: int,
                 kernel_size: List[int],
                 stride: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride

    def keys(self):
        return ['input_dim', 'output_dim', 'kernel_size', 'stride']

    def __getitem__(self, key):
        return getattr(self, key)


class CrossformerParams:
    def __init__(self, input_dim: int,
                 group: int,
                 n_layer: int):
        self.input_dim = input_dim
        self.group = group
        self.n_layer = n_layer

    def keys(self):
        return ['input_dim', 'group', 'n_layer']

    def __getitem__(self, key):
        return getattr(self, key)


class CrossformerUnit(nn.Module):
    def __init__(self, cross_scale_params: CrossScaleParams, transformer_params: CrossformerParams):
        super(CrossformerUnit, self).__init__()
        self.encoder_embedding = CrossScaleEmbedding(
            **dict(cross_scale_params))
        self.encoder_layers = CrossformerPack(**dict(transformer_params))

        cross_scale_params.input_dim, cross_scale_params.output_dim = cross_scale_params.output_dim, cross_scale_params.input_dim
        transformer_params.input_dim = cross_scale_params.output_dim

        self.decoder_embedding = CrossScaleEmbedding(
            **cross_scale_params, reversed=True)
        self.decoder_layers = CrossformerPack(**dict(transformer_params))

    def forward(self, x, next: List[nn.Module]):
        h = self.encoder_embedding(x)
        h = self.encoder_layers(h)

        if len(next) > 0:
            first, rest = next[0], next[1:]
            h_next = first(h, rest)
        else:
            h_next = h

        h = self.decoder_embedding([h_next, h, x.shape])
        y = self.decoder_layers(h)
        return y


class CrossformerEncoder(nn.Module):
    def __init__(self, cross_params: List[CrossScaleParams], crossformer_params: List[CrossformerParams]):
        super(CrossformerEncoder, self).__init__()
        assert len(cross_params) == len(crossformer_params)
        self.embeddings = nn.ModuleList()
        self.crossformers = nn.ModuleList()

        for cross_param in cross_params:
            cross_param = dict(cross_param)
            self.embeddings.append(CrossScaleEmbedding(**cross_param))

        for crossformer_param in crossformer_params:
            crossformer_param = dict(crossformer_param)
            self.crossformers.append(CrossformerPack(**crossformer_param))

    def forward(self, x):
        features = []
        shapes = []
        for embedding, crossformer in zip(self.embeddings, self.crossformers):
            shapes.append(x.shape)
            x = embedding(x)
            x = crossformer(x)
            features.append(x)
        return features, shapes


class CrossformerDecoder(nn.Module):
    def __init__(self, cross_params: List[CrossScaleParams], crossformer_params: List[CrossformerParams]):
        super(CrossformerDecoder, self).__init__()
        assert len(cross_params) == len(crossformer_params)
        self.embeddings = nn.ModuleList()
        self.crossformers = nn.ModuleList()

        for cross_param in cross_params:
            cross_param = dict(cross_param)
            self.embeddings.append(CrossScaleEmbedding(
                **cross_param, reversed=True))

        for crossformer_param in crossformer_params:
            crossformer_param = dict(crossformer_param)
            self.crossformers.append(CrossformerPack(**crossformer_param))

    def forward(self, features: List, shapes: List):
        assert len(features) == len(shapes)
        assert len(features) == len(self.embeddings)
        features = list(reversed(features))
        shapes = list(reversed(shapes))
        y = features[0]
        for buddy, shape, embedding, crossformer in zip(features, shapes, self.embeddings, self.crossformers):
            y = embedding([buddy, y, shape])
            y = crossformer(y)
        return y
