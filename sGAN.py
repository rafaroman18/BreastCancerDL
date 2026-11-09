#  In this file we are going to create a semisupervised GAN
import pandas as pd
from constants import *
from datasets.cancer_detection_dataset import CancerDetectionDataset
import torch
import torch.nn as nn
from torchvision.models import resnet18

# Weights Initialization
def weights_init_normal(m):
    """
    This function will initialize the weights of the layer of our models.
    params:
        m: the model to initialize: Generator or Discriminator
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Creating Generator and the Discriminator with 2 blocks:
# the supervised and unsupervised

## Generator with Transfer Learning
class Generator_TF(nn.Module):
    def __init__(self):
        super(Generator_TF, self).__init__()
        
        # First layers
        self.input_layers = nn.Sequential(
            nn.Linear(LATENT_DIM, 224))

        # Creating ResNet network
        self.resnet = resnet18(pretrained=True)
        
        # Freezing the model
        for param in self.resnet.parameters():
            param.requires_grad = False 
        
        # Output layers
        self.linear_layer = nn.Sequential(nn.Linear(1000, 64 * 14 * 14))
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=2), 
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Tanh()
        )
        

    def forward(self, noise):
        img = self.resnet(noise)
        img = self.linear_layer(img)
        img = img.view(img.shape[0], 64, 14, 14)
        img = self.conv_blocks(img)
        return img
    
    def __str__(self):
        return "sGAN - Generator"
    
    
## Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # First layers
        self.input_layers = nn.Sequential(
            nn.Linear(3 * 7 * 7, 64 * 14 * 14))
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=2), 
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Tanh()
        )
        

    def forward(self, noise):
        noise_reshaped = noise.reshape(noise.shape[0], 3*7*7)
        img = self.input_layers(noise_reshaped)
        img = img.view(img.shape[0], 64, 14, 14)
        img = self.conv_blocks(img)
        return img
    
    def __str__(self):
        return "sGAN - Generator"


## Discriminator
## As we said, we will create a discriminator with 2 blocks:
## supervised and unsueprvised
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True)
                     ,nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
    
        # Conv blocks
        self.conv_blocks = self.conv_blocks = nn.Sequential(
            *discriminator_block(CONST.DATA_CHANNELS, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        ## Output layers        
        # Adversarial layer (fake/real)
        self.adversarial_layer = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())
        # Classifier layer (classifier)
        self.classifier_layer = nn.Sequential(nn.Linear(4096, CONST.NUM_CLASSES_SGAN)) 


    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], 4096) 
        generated = self.adversarial_layer(out)
        class_label = self.classifier_layer(out)

        return generated, class_label

    def __str__(self):
        return "sGAN - Discriminator"