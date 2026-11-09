#  In this file we are going to create a conditional GAN that generate
# images and use that images to expand our dataset
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
        
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        
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
            nn.ConvTranspose2d(64, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=2), 
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(1024, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1024, 1, kernel_size=4, stride=1, padding=1),
            nn.Tanh()
        )
        

    def forward(self, noise, labels):        
        img = self.resnet(noise)
        img = self.linear_layer(img)
        img = img.view(img.shape[0], 64, 14, 14)
        img = self.conv_blocks(img)
        return img
    
    def __str__(self):
        return "cGAN - Generator"
    
    
## Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        
        # Input layer
        self.input_layers = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 64 * 14 * 14)) # 100 + 2 (LATENT_DIM + EMBEDDING VECTORS)
        
        # Conv layers
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=2), 
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(1024, 0.8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1024, 1, kernel_size=4, stride=1, padding=1),
            nn.Tanh()
        )
        

    def forward(self, noise, labels):
        # Getting embedding vector
        c = self.label_emb(labels)
        
        # Concatenating image with label
        noise_ = torch.cat([noise, c], 1)
        
        noise_reshaped = noise_.reshape(noise.shape[0], LATENT_DIM + NUM_CLASSES) # 100 + 2 (LATENT_DIM + EMBEDDING VECTORS)
        img = self.input_layers(noise_reshaped)
        img = img.view(img.shape[0], 64, 14, 14)
        img = self.conv_blocks(img)
        return img
    
    def __str__(self):
        return "cGAN - Generator"


## Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Embed labels
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        
        # Input Layer (linear)
        self.input_layer = nn.Linear((56 * 56) + 2, 1 * 64 * 64)
        
        # Intermediate Layer (linear)
        self.intermediate_layer = nn.Linear(4, 4096)

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
            *discriminator_block(DATA_CHANNELS, 1024, bn=False),
            *discriminator_block(1024, 512),
            *discriminator_block(512, 256),
            *discriminator_block(256, 1),
        )

        # Output layer
        self.output_layer = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, img, labels):
        
        # Transform image into (1, BATCH_SIZE * IMG_SIZE * IMG_SIZE) vector
        img_ = img.view(img.size()[0], IMG_SIZE * IMG_SIZE)
        # Embed labels
        c = self.label_emb(labels)
        # Concat images and labels
        img_ = torch.cat([img_, c], 1)
        
        # Linear Input Layer
        img_ = self.input_layer(img_)
        
        # Changing Shape
        img_ = img_.view(img.size()[0], 1, 64, 64)
        
        # Conv Blocks
        img_ = self.conv_blocks(img_)
        
        # Changing Output Shape
        img_ = img_.view(img.size()[0], 16)
        
        # Output Layer
        out = self.output_layer(img_)

        return out

    def __str__(self):
        return "cGAN - Discriminator"