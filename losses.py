"""
Implements the paper's loss functions for training its models.
"""

import torch


class DiscriminatorReconstructionLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, real_images, decoded_images):
        return torch.mean(torch.abs(real_images - decoded_images))


class DiscriminatorRealFakeLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, real_fake_output_logits_on_real_images,
                real_fake_output_logits_on_fake_images):
        real_loss = torch.mean(
            torch.nn.ReLU(1 - real_fake_output_logits_on_real_images))

        fake_loss = torch.mean(
            torch.nn.ReLU(1 + real_fake_output_logits_on_fake_images))

        return real_loss + fake_loss


class GeneratorLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, real_fake_output_logits_on_fake_images):
        return -1 * torch.mean(real_fake_output_logits_on_fake_images)
