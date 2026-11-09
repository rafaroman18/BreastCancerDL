import torchvision
from torchview import draw_graph
from models.styleGAN import Generator, Discriminator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

noise = torch.randn(64, 512).to(device)

gen = Generator(512, 512, 512, device).to(device)
out = gen(noise, 1e-7, 5)

model_graph = draw_graph(Generator(512, 512, 512, device), input_data=(noise, 1e-7, 5), expand_nested=True, save_graph=True, filename='docs/assets/stylegan_generator_structure')

