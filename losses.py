"""
Implements the paper's loss functions for training its models.
"""

import torch

def discriminator_reconstruction_loss(real_images, decoded_images):
    return torch.mean(torch.abs(real_images - decoded_images))

def discriminator_real_fake_loss(real_fake_output_logits_on_real_images,
                                 real_fake_output_logits_on_fake_images):
    real_loss = torch.mean(
        torch.nn.ReLU(1 - real_fake_output_logits_on_real_images))

    fake_loss = torch.mean(
        torch.nn.ReLU(1 + real_fake_output_logits_on_fake_images))

    return real_loss + fake_loss

def generator_loss(real_fake_output_logits_on_fake_images):
    return -1 * torch.mean(real_fake_output_logits_on_fake_images)
