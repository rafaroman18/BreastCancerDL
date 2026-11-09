import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log2
import numpy as np
import os
from constants import CONST
from torchvision.models import resnet18


# Copyright [2022] [TAUIL Abd Elilah]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


######### Layers and Blocks Definitions
# Convolutional layer
class SGConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SGConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

    
# Mapping Network Linear class
class SGLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SGLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2/in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None
        
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        return self.linear(x * self.scale) + self.bias
    
    
# Generator Block
# We will define the 2 convolutional layers, 2 inject Noise layers 
# and 2 AdaIN layers that conforms the Synthesis Network
class GenBlock(nn.Module):
    def __init__(self, in_channel, out_channel, w_dim, device):
        super(GenBlock, self).__init__()
        self.conv1 = SGConv2D(in_channel, out_channel)
        self.conv2 = SGConv2D(out_channel, out_channel)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = injectNoise(out_channel, device)
        self.inject_noise2 = injectNoise(out_channel, device)
        self.adain1 = AdaIN(out_channel, w_dim)
        self.adain2 = AdaIN(out_channel, w_dim)
        
    def forward(self, x, w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        
        return x

###### Mapping Network Definitions ######
# Normalization of pixels noise
class PixenNorm(nn.Module):
    def __init__(self):
        super(PixenNorm, self).__init__()
        self.epsilon = 1e-8
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

# Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixenNorm(),
            SGLinear(z_dim, w_dim),
            nn.ReLU(),
            SGLinear(w_dim, w_dim),
            nn.ReLU(),
            SGLinear(w_dim, w_dim),
            nn.ReLU(),
            SGLinear(w_dim, w_dim),
            nn.ReLU(),
            SGLinear(w_dim, w_dim),
            nn.ReLU(),
            SGLinear(w_dim, w_dim),
            nn.ReLU(),
            SGLinear(w_dim, w_dim),
            nn.ReLU(),
            SGLinear(w_dim, w_dim),
        )
    
    def forward(self, x):
        return self.mapping(x)
    
###### StyleGAN Generator (Synthesis Network) Definitions ######
# AdaIN: instance normalization
class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.scale = SGLinear(w_dim, channels)
        self.bias = SGLinear(w_dim, channels)
        
    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_scale
    
# Inserting noise in the synthesis network
class injectNoise(nn.Module):
    def __init__(self, channels, device):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.device = device
        
    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=self.device)
        return x + self.weight + noise

###### Generator Definitions ######
class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, in_channels, device):
        super().__init__()
        self.starting_cte = nn.Parameter(torch.ones(1, in_channels, 4, 4))
        self.map = MappingNetwork(z_dim, w_dim)
        self.initial_adain1 = AdaIN(in_channels, w_dim)
        self.initial_adain2 = AdaIN(in_channels, w_dim)
        self.initial_noise1 = injectNoise(in_channels, device)
        self.initial_noise2 = injectNoise(in_channels, device)
        self.initial_conv   = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky          = nn.LeakyReLU(0.2, inplace=True)

        self.initial_4x4   = SGConv2D(
            in_channels, 1, kernel_size = 1, stride=1, padding=0
        )
        self.prog_blocks, self.layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_4x4])
        )

        for i in range(len(CONST.FACTORS)-1):
            conv_in_c  = int(in_channels * CONST.FACTORS[i])
            conv_out_c = int(in_channels * CONST.FACTORS[i+1])
            self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, w_dim, device))
            self.layers.append(SGConv2D(conv_out_c, 1, kernel_size = 1, stride=1, padding=0))
        
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha ) * upscaled)

    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        x = self.initial_adain1(self.initial_noise1(self.starting_cte),w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

        if steps == 0:
            return self.initial_4x4(x)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode = 'bilinear', align_corners=False)
            out      = self.prog_blocks[step](upscaled,w)

        final_upscaled = self.layers[steps-1](upscaled)
        final_out      = self.layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)
    
    def __str__(self):
        return "StyleGAN - Generator"
    
    
###### Discriminator Definitions ######
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # We take factors from the end to the beginning because 
        # the Discriminator should be mirrored from the Generator
        for i in range(len(CONST.FACTORS) - 1, 0, -1):
            conv_in = int(in_channels * CONST.FACTORS[i])
            conv_out = int(in_channels * CONST.FACTORS[i - 1])
            
            # Adding Convolutional Blocks
            self.prog_blocks.append(
                 nn.Sequential(
                     SGConv2D(conv_in, conv_out),
                     nn.LeakyReLU(0.2),
                     SGConv2D(conv_out, conv_out),
                     nn.LeakyReLU(0.2),
                 )
            )

            self.layers.append(
                SGConv2D(1, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # This layer is the mirror of the first Generator Layer (initial_4x4)
        self.final_4x4 = SGConv2D(
            1, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.layers.append(self.final_4x4)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  
        
        # Final Block (4x4)
        self.final_block = nn.Sequential(
            
            SGConv2D(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            SGConv2D(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            
            # Due to the fact we are implementing StyleGAN and the 
            # function loss is Wasserstein Loss, the output layer is 
            # Convolutional 2D not Sigmoid as usual
            SGConv2D(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            )  
        
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def std_batch(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps

        out = self.leaky(self.layers[cur_step](x))

        if steps == 0: 
            out = self.std_batch(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.std_batch(out)
        return self.final_block(out).view(out.shape[0], -1)
    
    def __str__(self):
        return "StyleGAN - Discriminator"