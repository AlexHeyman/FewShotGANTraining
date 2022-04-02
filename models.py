"""
Implements the models in the paper's experiments as PyTorch Modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipLayerExcitation(nn.Module):

    def __init__(self, in_channels_1, in_channels_2):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.conv1 = nn.Conv2d(in_channels_2, in_channels_2,
                               kernel_size=4, stride=1, padding=0)
        self.nonlinear1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(in_channels_2, in_channels_1,
                               kernel_size=1, stride=1, padding=0)
        self.nonlinear2 = nn.Sigmoid()

    def forward(self, x1, x2):
        x2 = self.pooling(x2)
        x2 = self.nonlinear1(self.conv1(x2))
        x2 = self.nonlinear2(self.conv2(x2))
        return x1 * x2


class Upsampler(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels * 2,
                              kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels * 2)
        self.glu = nn.GLU() # cuts number of channels in half

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.glu(x)
        return x


class Generator(nn.Module):

    def __init__(self, use_skips, output_resolution):
        # output_resolution is assumed to be either 256 or 1024
        super().__init__()
        self.use_skips = use_skips
        self.output_resolution = output_resolution
        self.conv_transpose = nn.ConvTranspose2d(256, 2048, kernel_size=4,
                                                 stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(2048)
        self.glu = nn.GLU() # cuts number of channels in half
        self.upsampler_4 = Upsampler(1024, 512)
        self.upsampler_8 = Upsampler(512, 256)
        self.upsampler_16 = Upsampler(256, 128)
        self.upsampler_32 = Upsampler(128, 128)
        self.upsampler_64 = Upsampler(128, 64)
        self.upsampler_128 = Upsampler(64, 32)
        self.upsampler_256 = Upsampler(32, 3)
        self.upsampler_512 = Upsampler(3, 3)
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.nonlinear = nn.Tanh()
        if use_skips:
            self.skip_8_128 = SkipLayerExcitation(64, 512)
            self.skip_16_256 = SkipLayerExcitation(32, 256)
            self.skip_32_512 = SkipLayerExcitation(3, 128)

    def forward(self, x):
        x_4 = self.glu(self.batchnorm(self.conv_transpose(x)))
        x_8 = self.upsampler_4(x_4)
        x_16 = self.upsampler_8(x_8)
        x_32 = self.upsampler_16(x_16)
        x_64 = self.upsampler_32(x_32)

        if self.output_resolution == 256:
            if self.use_skips:
                x_128 = self.skip_8_128(self.upsampler_64(x_64), x_8)
                x_penultimate = self.skip_16_256(
                    self.upsampler_128(x_128), x_16)
            else:
                x_128 = self.upsampler_64(x_64)
                x_penultimate = self.upsampler_128(x_128)
        else:
            if self.use_skips:
                x_128 = self.skip_8_128(self.upsampler_64(x_64), x_8)
                x_256 = self.skip_16_256(self.upsampler_128(x_128), x_16)
                x_512 = self.skip_32_512(self.upsampler_256(x_256), x_32)
            else:
                x_128 = self.upsampler_64(x_64)
                x_256 = self.upsampler_128(x_128)
                x_512 = self.upsampler_256(x_256)
            x_penultimate = self.upsampler_512(x_512)

        return self.nonlinear(self.conv(x_penultimate))


class SimpleDecoder(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.upsampler1 = Upsampler(in_channels, 128)
        self.upsampler2 = Upsampler(128, 64)
        self.upsampler3 = Upsampler(64, 32)
        self.upsampler4 = Upsampler(32, 3)

    def forward(self, x):
        x = self.upsampler1(x)
        x = self.upsampler2(x)
        x = self.upsampler3(x)
        x = self.upsampler4(x)
        return x


class ResidualDownsampler(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.nonlinear1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.nonlinear2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.nonlinear3 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x1 = self.nonlinear1(self.batchnorm1(self.conv1(x)))
        x2 = self.nonlinear2(self.batchnorm2(self.conv2(x)))
        x3 = self.nonlinear3(self.batchnorm3(self.conv3(x2)))
        return x1 + x3


class Discriminator(nn.Module):

    def __init__(self, use_decoders, input_resolution):
        # input_resolution is assumed to be either 256 or 1024
        super().__init__()
        self.use_decoders = use_decoders
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.nonlinear1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.nonlinear2 = nn.LeakyReLU(negative_slope=0.1)
        
        self.downsampler_256 = ResidualDownsampler(32, 64)
        self.downsampler_128 = ResidualDownsampler(64, 128)
        self.downsampler_64 = ResidualDownsampler(128, 256)
        self.downsampler_32 = ResidualDownsampler(256, 256)
        self.downsampler_16 = ResidualDownsampler(256, 256)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.nonlinear3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0)

        if use_decoders:
            self.decoder1 = SimpleDecoder()
            self.decoder2 = SimpleDecoder()

    def forward(self, x):
        x1 = self.nonlinear1(self.conv1(x))
        x_256 = self.nonlinear2(self.batchnorm2(self.conv2(x1)))
        
        x_128 = self.downsampler_256(x_256)
        x_64 = self.downsampler_128(x_128)
        x_32 = self.downsampler_64(x_64)
        x_16 = self.downsampler_32(x_32)
        x_8 = self.downsampler_16(x_16)

        x_penultimate = self.nonlinear3(self.batchnorm3(self.conv3(x_8)))
        logits = self.conv4(x_penultimate)
        
        if self.use_decoders:
            I = F.interpolate(x, scale_factor=0.5, mode='bilinear')

            ##### Change later once I figure out how the cropping works
            I_part = F.interpolate(x, size=(128, 128), mode='bilinear')
            x_16_part = F.interpolate(x_16, size=(8, 8), mode='bilinear')
            #####

            I_prime = self.decoder2(x_8)
            I_part_prime = self.decoder1(x_16)
            
            return logits, I, I_part, I_prime, I_part_prime
        else:
            return logits
