"""
Implements the models in the paper's experiments as PyTorch Modules.
"""

from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipLayerExcitation(nn.Module):
    """
    The "skip-layer excitation module" used by the Generator to create skip
    connections between layers.
    """

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
        # x1 must have shape ([batch_size], in_channels_1, [height1], [width1])
        # x2 must have shape ([batch_size], in_channels_2, [height2], [width2])
        # output has shape ([batch_size], in_channels_1, [height1], [width1])
        x2 = self.pooling(x2)
        x2 = self.nonlinear1(self.conv1(x2))
        x2 = self.nonlinear2(self.conv2(x2))
        return x1 * x2


class Upsampler(nn.Module):
    """
    An up-sampling layer used by both the Generator and the SimpleDecoder.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels * 2,
                              kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels * 2)
        self.glu = nn.GLU() # cuts number of channels in half

    def forward(self, x):
        # x must have shape ([batch_size], in_channels, [height], [width])
        # output has shape ([batch_size], out_channels, [height]*2, [width]*2)
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.glu(x)
        return x


class Generator(nn.Module):
    """
    The GAN's generator.
    """

    def __init__(self, use_skips, output_resolution):
        # output_resolution should be either 256 or 1024
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
        self.conv = nn.Conv2d(32 if output_resolution == 256 else 3, 3,
                              kernel_size=3, stride=1, padding=1)
        self.nonlinear = nn.Tanh()
        if use_skips:
            self.skip_8_128 = SkipLayerExcitation(64, 512)
            self.skip_16_256 = SkipLayerExcitation(32, 256)
            self.skip_32_512 = SkipLayerExcitation(3, 128)

    def forward(self, x):
        # x is a set of random noise vectors with shape ([batch_size], 256)
        # output is a set of images with shape
        # ([batch_size], 3, output_resolution, output_resolution)
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
    """
    A decoder used to help train the Discriminator.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.upsampler1 = Upsampler(in_channels, 256)
        self.upsampler2 = Upsampler(256, 128)
        self.upsampler3 = Upsampler(128, 128)
        self.upsampler4 = Upsampler(128, 3)

    def forward(self, x):
        # x must be of shape ([batch_size], in_channels, [height], [width])
        # output is of shape ([batch_size], 3, [height] * 16, [width] * 16)
        x = self.upsampler1(x)
        x = self.upsampler2(x)
        x = self.upsampler3(x)
        x = self.upsampler4(x)
        return x


class ResidualDownsampler(nn.Module):
    """
    A down-sampling structure used by the Discriminator.
    """

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
        # x must have shape ([batch_size], in_channels, [height], [width])
        # output has shape ([batch_size], out_channels, [height/2], [width/2])
        x1 = self.nonlinear1(self.batchnorm1(self.conv1(x)))
        x2 = self.nonlinear2(self.batchnorm2(self.conv2(x)))
        x3 = self.nonlinear3(self.batchnorm3(self.conv3(x2)))
        return x1 + x3


class Discriminator(nn.Module):
    """
    The GAN's discriminator.
    """

    def __init__(self, use_decoders, input_resolution):
        # input_resolution should be either 256 or 1024
        super().__init__()
        self.use_decoders = use_decoders
        self.input_resolution = input_resolution
        
        early_conv_filters = (8 if input_resolution == 256 else 32)
        early_conv_stride = (1 if input_resolution == 256 else 2)
        self.conv1 = nn.Conv2d(3, early_conv_filters, kernel_size=4,
                               stride=early_conv_stride, padding='same')
        self.nonlinear1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(early_conv_filters, early_conv_filters,
                               kernel_size=4, stride=early_conv_stride,
                               padding='same')
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.nonlinear2 = nn.LeakyReLU(negative_slope=0.1)
        
        self.downsampler_256 = ResidualDownsampler(32, 64)
        self.downsampler_128 = ResidualDownsampler(64, 128)
        self.downsampler_64 = ResidualDownsampler(128, 128)
        self.downsampler_32 = ResidualDownsampler(128, 256)
        self.downsampler_16 = ResidualDownsampler(256, 512)

        self.conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.nonlinear3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)

        if use_decoders:
            self.decoder1 = SimpleDecoder(256)
            self.decoder2 = SimpleDecoder(512)

    def forward(self, x, label):
        # x must be a set of images with shape
        # ([batch_size], 3, input_resolution, input_resolution)
        # label should be either 0 (fake) or 1 (real)
        # output is a set of real/fake logits with shape ([batch_size], 5, 5)
        x1 = self.nonlinear1(self.conv1(x))
        x_256 = self.nonlinear2(self.batchnorm2(self.conv2(x1)))
        
        x_128 = self.downsampler_256(x_256)
        x_64 = self.downsampler_128(x_128)
        x_32 = self.downsampler_64(x_64)
        x_16 = self.downsampler_32(x_32)
        x_8 = self.downsampler_16(x_16)

        x_penultimate = self.nonlinear3(self.batchnorm3(self.conv3(x_8)))
        logits = self.conv4(x_penultimate)
        
        if self.use_decoders and label == 1:
            
            I = F.interpolate(x, size=(128, 128), mode='bilinear')

            # Random cropping
            if self.input_resolution != 256:
                x = F.interpolate(x, size=(256, 256), mode='bilinear')
            crop_area = randint(0, 3)
            if crop_area == 0: # Top left
                I_part = x[:, :, :128, :128]
                x_16_part = x_16[:, :, :8, :8]
            elif crop_area == 1: # Bottom left
                I_part = x[:, :, 128:, :128]
                x_16_part = x_16[:, :, 8:, :8]
            elif crop_area == 2: # Top right
                I_part = x[:, :, :128, 128:]
                x_16_part = x_16[:, :, :8, 8:]
            else: # Bottom right
                I_part = x[:, :, 128:, 128:]
                x_16_part = x_16[:, :, 8:, 8:]

            I_prime = self.decoder2(x_8)
            I_part_prime = self.decoder1(x_16)
            
            return logits, I, I_part, I_prime, I_part_prime
        else:
            return logits
