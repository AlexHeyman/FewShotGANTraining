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
output_folder = 'outputs/'


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
eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)
eval_images = torch.concat([image for image in eval_loader], dim=0)


# This noise tensor will be used on each GAN to generate a set of images to be
# compared against the original dataset
noise = torch.Tensor(images_to_generate, 256).normal_(0, 1).to(device)

# Create a view of the noise tensor as a list of row vectors so they can be
# fed into a GAN's generator one at a time, prevening out-of-memory errors
noise_segments = [torch.reshape(noise[i], (1, 256))
                  for i in range(images_to_generate)]


def evaluate(ts):
    ts.load_checkpoint(directory_path=checkpoints_folder,
                       filename=('%s_%d.pt' % (ts.name, iteration_to_eval)))
    
    fake_images = torch.concat([ts.generator(noise_segments[i])
                                for i in range(images_to_generate)], dim=0)

    # Calculate FID between generated images and original dataset
    fid = FID(fake_images, eval_images)
    print('%s FID: %f' % (ts.name, fid.item()))

    # Plot 25 of the generated images in a 5x5 grid
    images_to_plot = torch.clone(fake_images[0:25])
    images_to_plot = images_to_plot / 2 + 0.5 # Unnormalize
    plt.imshow(np.transpose(images_to_plot.numpy(), (1, 2, 0)),
               nrow=5, padding=10)
    plt.savefig(path.join(output_folder, ('samples_%s.png' % ts.name)))


# Evaluate Skip + Decode
evaluate(GANTrainingSystem(
    'skipdecode', True, True, resolution, eval_loader, device))


# Evaluate Skip alone
evaluate(GANTrainingSystem(
    'skip', True, False, resolution, eval_loader, device))


# Evaluate Decode alone
evaluate(GANTrainingSystem(
    'decode', False, True, resolution, eval_loader, device))


# Evaluate baseline model
evaluate(GANTrainingSystem(
    'baseline', False, False, resolution, eval_loader, device))
