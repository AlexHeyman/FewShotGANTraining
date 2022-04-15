"""
A script for training the four variants of the paper's GAN model as in the
"Few-shot generation" experiment.
"""

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from image_folder_dataset import ImageFolderDataset
from gan_training_system import GANTrainingSystem


# Resolution of input and output images (both height and width)
# Training dataset images will be resized to match if necessary
resolution = 256

iterations_to_run = 50000
print_losses_at_end = True
dataset_folder = './images/AnimalFace-dog/img/'
checkpoints_folder = 'checkpoints/'


# Do the training on the GPU if we can, and on the CPU otherwise
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print('Using device: %s' % device)


transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

trainset = ImageFolderDataset(root=dataset_folder, transform=transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4,
                         pin_memory=True)


# Train with Skip + Decode
ts = GANTrainingSystem('skipdecode', True, True, resolution, trainloader, device)
ts.train(iterations_to_run, print_losses_at_end, checkpoints_folder)


# Train with Skip alone
ts = GANTrainingSystem('skip', True, False, resolution, trainloader, device)
ts.train(iterations_to_run, print_losses_at_end, checkpoints_folder)


# Train with Decode alone
ts = GANTrainingSystem('decode', False, True, resolution, trainloader, device)
ts.train(iterations_to_run, print_losses_at_end, checkpoints_folder)


# Train baseline model
ts = GANTrainingSystem('baseline', False, False, resolution, trainloader, device)
ts.train(iterations_to_run, print_losses_at_end, checkpoints_folder)
