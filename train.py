"""
A script for training the four variants of the paper's GAN model as in the
"Few-shot generation" experiment.
"""

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from image_folder_dataset import ImageFolderDataset
from gan_training_system import GANTrainingSystem


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print('Using device: %s' % device)


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

trainset = ImageFolderDataset(root='./images/pokemon/img/', transform=transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8,
                        pin_memory=True)


ts = GANTrainingSystem('test', True, True, 256, trainloader, device)
ts.train(100)
