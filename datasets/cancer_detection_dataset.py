import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
import random


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CancerDetectionDataset(Dataset):
    # This is our dataset class which mix the data from a csv file (labels)
    # and the dataset folder (images) to generate a CancerDetectionDataset variable.
    # We will also add some functions so we can return labeled and unlabeled data 
    # params:
        # path_images_dir: path of the dataset
        # df: path of the csv file (labels)
    
    def __init__(self, path_images_dir, df):
        self.path_images_dir = path_images_dir
        self.transform = transform
        self.samples = []
        self.groups = []

        for index, row in df.iterrows():
            image_path = f"{path_images_dir}/{row['Folder']}/{row['Image path']}"
            label = 0 if row['Diagnosis'] == 'Healthy' else 1
            sample = (image_path, label)
            self.samples.append(sample)
            self.groups.append(row['Folder'])
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, image_path
    
    def subset(self, ids):
 
        subset_dicts = []
        
        # Iterating through the given ids 
        for i in ids:
            subset_dicts.append({'Diagnosis': 'Sick' if (self.samples[i][1] == 1) else 'Healthy',
                              'Type': 'Static',
                              'Folder': self.groups[i],
                              'Image path': self.samples[i][0].replace(self.path_images_dir + '/' + self.groups[i], '').replace('/', '')})
            
        # Creating the Dataframe
        df_subset = pd.DataFrame(subset_dicts)
        
        # Returning the dataset variable
        return CancerDetectionDataset(self.path_images_dir, df_subset)
            
           
    
    