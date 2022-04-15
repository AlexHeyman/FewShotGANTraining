"""
A script for evaluating the synthesis performance of the four variants of the
paper's GAN model.
"""

from os import path
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from image_folder_dataset import ImageFolderDataset
from gan_training_system import GANTrainingSystem
from metrics import FID


# Resolution of input and output images (both height and width)
# Training dataset images will be resized to match if necessary
resolution = 256

iteration_to_eval = 50000
images_to_generate = 5000
dataset_folder = './images/AnimalFace-dog/img/'
checkpoints_folder = 'checkpoints/'


# Do the evaluation on the GPU if we can, and on the CPU otherwise
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print('Using device: %s' % device)


transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

eval_set = ImageFolderDataset(root=dataset_folder, transform=transform)
eval_loader = DataLoader(eval_set, batch_size=len(eval_set), shuffle=False,
                         num_workers=4, pin_memory=True)
eval_images = next(iter(eval_loader))


# This noise vector will be used on each GAN to generate a set of images to be
# compared against the original dataset
noise = torch.Tensor(images_to_generate, 256).normal_(0, 1).to(self.device)


def evaluate(ts):
    ts.load_checkpoint(directory_path=checkpoints_folder,
                       filename='%s_%d.pt' % (ts.name, iteration_to_eval))
    fake_images = self.generator(noise)

    # Calculate FID between generated images and original dataset
    fid = FID(fake_images, eval_images)
    print('%s FID: %f' % (ts.name, fid.item()))

    # Plot 25 of the generated images in a 5x5 grid
    images_to_plot = torch.clone(fake_images[0:25])
    images_to_plot = images_to_plot / 2 + 0.5 # Unnormalize
    plt.imshow(np.transpose(images_to_plot.numpy(), (1, 2, 0)),
               nrow=5, padding=10)
    plt.savefig(path.join(checkpoints_folder, ('%s.png' % ts.name)))


# Evaluate Skip + Decode
evaluate(GANTrainingSystem(
    'skipdecode', True, True, resolution, trainloader, device))


# Evaluate Skip alone
evaluate(GANTrainingSystem(
    'skip', True, False, resolution, trainloader, device))


# Evaluate Decode alone
evaluate(GANTrainingSystem(
    'decode', False, True, resolution, trainloader, device))


# Evaluate baseline model
evaluate(GANTrainingSystem(
    'baseline', False, False, resolution, trainloader, device))
