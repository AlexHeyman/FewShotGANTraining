"""
Implements the paper's metrics for evaluating models' synthesis performance.
"""

import torch

def FID(images1, images2):
    """
    Returns the Frechet Inception Distance between two sets of images.

    images1 and images2 are both PyTorch Tensors with shape (number_of_images,
    image_height, image_width). number_of_images need not be the same for both,
    but image_height and image_width must be the same.
    """

    images1 = torch.flatten(images1, start_dim=1)
    images2 = torch.flatten(images2, start_dim=1)

    mean1 = torch.mean(images1, dim=0)
    mean2 = torch.mean(images2, dim=0)

    cov1 = torch.cov(images1)
    cov2 = torch.cov(images2)

    d_squared = torch.square(torch.dist(mean1, mean2, p=2))\
                + torch.trace(cov1 + cov2\
                              - 2 * torch.sqrt(torch.matmul(cov1, cov2)))
    return torch.sqrt(d_squared)
