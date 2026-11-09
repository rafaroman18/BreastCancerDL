import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision import transforms
import random
from torchvision.models import resnet18
from constants import *
import torch

means = [0.865, 0.885, 0.848, 0.838, 0.839]
variance = [0.105, 0.003, 0.004, 0.006, 0.001]

x_gen = ['0%', '~25%', '~50%', '~75%', '~100%']
y_time = [66, 5490, 5584, 5873, 5883]

fig, ax1 = plt.subplots()

## Loss
# Adjusting the epochs
epochs = np.arange(0, (len(CONST.BATCH_SIZES)*CONST.EPOCHS) + 1, CONST.KID_SCORE_CHECKPOINT)

ax1.set_xlabel('Imágenes generadas (%)')
ax1.set_ylabel('Tiempo de ejecución (mins)')

# Putting the title and legend
plt.title('Tiempo de ejecución de las evaluaciones realizadas.')

ax1.grid()

plt.plot(x_gen, y_time, '-o', color='green')

ax1.annotate('66', xy=(0, 0), xytext=(0.1, 68), color='green')
ax1.annotate('5490', xy=(1, 5490), xytext=(1.1, 5200), color='green')
ax1.annotate('5584', xy=(2, 5584), xytext=(2.1, 5250), color='green')
ax1.annotate('5873', xy=(3, 5873), xytext=(3.1, 5500), color='green')
ax1.annotate('5883', xy=(4, 5883), xytext=(3.8, 5500), color='green')

plt.savefig('docs/assets/execution_time_plot.png')