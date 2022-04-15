"""
Defines a wrapper class responsible for creating an instance of the paper's GAN
model, training it, and saving and loading checkpoints of it.
"""

from os import path
import time
import torch
from torch.optim import Adam
from models import Generator, Discriminator
from losses import DiscriminatorReconstructionLoss,\
     DiscriminatorRealFakeLoss, GeneratorLoss


def gan_init(m):
    """
    Initialization function for the GAN's trainable parameters.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GANTrainingSystem:

    def __init__(self, name, use_skips, use_decoders, resolution, trainloader,
                 device):
        self.name = name
        self.use_skips = use_skips
        self.use_decoders = use_decoders
        self.resolution = resolution
        self.trainloader = trainloader
        self.device = device

        self.learning_rate = 0.0002
        self.adam_betas = (0.5, 0.999)
        self.save_every = 1000
        self.print_every = 1000
        
        self.generator = Generator(use_skips, resolution).to(device)
        self.generator.apply(gan_init)
        self.gen_loss = GeneratorLoss().to(device)
        self.gen_opt = Adam(self.generator.parameters(),
                            lr=self.learning_rate, betas=self.adam_betas)
        
        self.discriminator = Discriminator(use_decoders, resolution).to(device)
        self.discriminator.apply(gan_init)
        self.disc_loss = DiscriminatorRealFakeLoss().to(device)
        if use_decoders:
            self.reconstruction_loss = DiscriminatorReconstructionLoss()
        self.disc_opt = Adam(self.discriminator.parameters(),
                             lr=self.learning_rate, betas=self.adam_betas)
        
        self.iterations_ran = 0
        self.running_generator_loss = 0
        self.running_discriminator_loss = 0
        self.running_reconstruction_loss = 0
        self.training_time = 0
  
    def save_checkpoint(self, directory_path='./', filename=None):
        if filename is None:
            filename = '%s_%d.pt' % (self.name, self.iterations_ran)
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'gen_opt_state_dict': self.gen_opt.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'disc_opt_state_dict': self.disc_opt.state_dict(),
            'iterations_ran': self.iterations_ran,
            'running_generator_loss': self.running_generator_loss,
            'running_discriminator_loss': self.running_discriminator_loss,
            'running_reconstruction_loss': self.running_reconstruction_loss,
            'training_time': self.training_time
            }, path.join(directory_path, filename))
  
    def load_checkpoint(self, directory_path='./', filename=None):
        if filename is None:
            filename = '%s_%d.pt' % (self.name, self.iterations_ran)
    
        checkpoint = torch.load(path.join(directory_path, filename))
        self.generator.load_state_dict(checkpoint['generator_state_dict'],
                                       strict=False)
        self.gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'],
                                           strict=False)
        self.disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
        self.iterations_ran = checkpoint['iterations_ran']
        self.running_generator_loss = checkpoint['running_generator_loss']
        self.running_discriminator_loss = checkpoint['running_discriminator_loss']
        self.running_reconstruction_loss = checkpoint['running_reconstruction_loss']
        self.training_time = checkpoint['training_time']
        
    def train(self, max_iteration, print_losses_at_end, save_directory_path='./'):
        if print_losses_at_end:
            gen_losses = []
            disc_losses = []
            recon_losses = []

        self.generator.train()
        self.discriminator.train()
        start_time = time.time()

        while self.iterations_ran < max_iteration:
            for real_images in self.trainloader:
                if self.iterations_ran % self.save_every == 0:
                    self.training_time += (time.time() - start_time)
                    
                    print('Training time up to iteration %d: %.3f seconds'\
                          % (self.iterations_ran, self.training_time))
                    self.save_checkpoint(directory_path=save_directory_path)
                    
                    start_time = time.time()
                
                real_images = real_images.to(self.device)
                noise = torch.Tensor(len(real_images), 256).normal_(0, 1)\
                        .to(self.device)
                fake_images = self.generator(noise)

                # Discriminator optimization step
                self.discriminator.zero_grad()

                if self.use_decoders:
                    real_logits, I, I_part, I_prime, I_part_prime\
                            = self.discriminator(real_images, label=1)
                    loss_r = self.reconstruction_loss(I, I_prime)\
                             + self.reconstruction_loss(I_part, I_part_prime)
                    loss_r.backward(retain_graph=True)
                else:
                    real_logits = self.discriminator(real_images, label=1)
                    loss_r = torch.Tensor([0])

                fake_logits = self.discriminator(fake_images.detach(), label=0)

                loss_d = self.disc_loss(real_logits, fake_logits)
                loss_d.backward()
                self.disc_opt.step()
                
                # Generator optimization step
                self.generator.zero_grad()
                fake_logits = self.discriminator(fake_images, label=0)
                loss_g = self.gen_loss(fake_logits)
                loss_g.backward()
                self.gen_opt.step()

                # Print running losses
                self.running_generator_loss += loss_g.item()
                self.running_discriminator_loss += loss_d.item()
                self.running_reconstruction_loss += loss_r.item()
                if self.iterations_ran % self.print_every == (self.print_every - 1):
                    self.training_time += (time.time() - start_time)

                    self.running_generator_loss /= self.print_every
                    self.running_discriminator_loss /= self.print_every
                    self.running_reconstruction_loss /= self.print_every
                    
                    print('Iteration %d losses: %.3f %.3f %.3f' %
                          (self.iterations_ran + 1,
                           self.running_generator_loss,
                           self.running_discriminator_loss,
                           self.running_reconstruction_loss))
                    
                    if print_losses_at_end:
                        gen_losses.append(self.running_generator_loss)
                        disc_losses.append(self.running_discriminator_loss)
                        recon_losses.append(self.running_reconstruction_loss)
                    
                    self.running_generator_loss = 0
                    self.running_discriminator_loss = 0
                    self.running_reconstruction_loss = 0

                    start_time = time.time()

                self.iterations_ran += 1
                if self.iterations_ran >= max_iteration:
                    break

        self.training_time += (time.time() - start_time)

        print('Training time up to iteration %d: %.3f seconds'\
                        % (self.iterations_ran, self.training_time))
        if self.iterations_ran % self.save_every == 0:
            self.save_checkpoint(directory_path=save_directory_path)

        if print_losses_at_end:
            print('Generator losses:', gen_losses)
            print('Discriminator losses:', disc_losses)
            print('Reconstruction losses:', recon_losses)
