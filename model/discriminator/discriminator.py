import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, config, gpu_list, *args, **kwargs):
        super(Discriminator, self).__init__()

        # Calculate output shape of image discriminator (PatchGAN)
        channels = 1

        def discriminator_block(in_filters, out_filters, normalize=True):  # 鉴别器块儿
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                padding=1)]  # layer += [conv + norm + relu]
            if normalize:  # 每次卷积尺寸会缩小一半，共卷积了4次
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # layer += [conv(3, 64) + relu]
            *discriminator_block(channels, 64, normalize=False),
            # layer += [conv(64, 128) + norm + relu]
            *discriminator_block(64, 128),
            # layer += [conv(128, 256) + norm + relu]
            *discriminator_block(128, 256),
            # layer += [conv(256, 512) + norm + relu]
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),  # layer += [pad]
            nn.Conv2d(512, 1, 4, padding=1)  # layer += [conv(512, 1)]
        )

    def forward(self, img):  # 输入(1, 3, 256, 256)
        return self.model(img)  # 输出(1, 1, 16, 16)
