import unittest
import torch
from model.user.net import UserNet
from thop import profile


class TestCrossFormer(unittest.TestCase):
    def test_crossformer(self):
        # TODO: Implement test for CrossFormer
        model = UserNet(None, None)
        data = torch.randn(1, 2, 224, 224)
        macs, params = profile(model, inputs=(data,))
        print(f"MACs: {macs / 1e9} G, Params: {params / 1e6} M")
