import torch
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from constants import CONST
import pandas as pd
from torch.utils.data import DataLoader
from datasets.cancer_detection_dataset import CancerDetectionDataset
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from math import log2
import warnings
from sklearn.model_selection import StratifiedGroupKFold


# KID Score always throw a Warning to inform about memory usage. We are going to ignore it
warnings.filterwarnings("ignore")



def not_leaking_data_augmentations(batch, p):
    """
    If p reaches the threshold, calling this function will apply data augmentations to all the images in the 
    """
    transforms_DA = transforms.Compose([
        transforms.ColorJitter(brightness=p),
        transforms.RandomHorizontalFlip(p=p),
    ])
    
    return transforms_DA(batch)
    
    

def data_augmentation(gen, val_dataset, num_imgs, sick_or_healthy, device, train_string, patience=3):
    """
    In this function we will generate the best images based on the KID Score obtained by the generated batch. If the batch reaches a KID Threshold, we will keep that images.
    """
    print(f'Starting Data Augmentation\n')
    imgs_generated = 0
    epochs_no_generation = 0 # We will save every batch even if the threshold its not reached once this variable hits the patience
    imgs_path = []
    folder = CONST.DATA_AUGMENTATION_SICK_FOLDER if (sick_or_healthy == 1) else CONST.DATA_AUGMENTATION_HEALTHY_FOLDER
    folder += train_string
    
    ## Creating the loader with the dataset given
    loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
                
    ## Looping until we get the number of images we want
    while imgs_generated < num_imgs:
        with torch.no_grad():
            
            # Iterating the loader
            for real_imgs, _, _ in loader:
                real_imgs = real_imgs.to(device)
                
                # Generating 64 images (a batch)
                noise = torch.randn(64, CONST.Z_DIM).to(device)    
                imgs = gen(noise, 1.0, len(CONST.BATCH_SIZES)-1)
                
                # Evaluating images by KID Score
                kid_batch, _ = evaluate_images(imgs, val_dataset, device)     
                # print(f'KID Score del batch actual: {kid_batch}. Threshold: {CONST.DATA_AUGMENTATION_THRESHOLD}.')
                
                # If it is a good batch, we save it
                if ((kid_batch < CONST.DATA_AUGMENTATION_THRESHOLD) & (kid_batch > - CONST.DATA_AUGMENTATION_THRESHOLD)) | (epochs_no_generation >= patience):
                    if not os.path.exists(f'./images/{folder}/'):
                            os.makedirs(f'./images/{folder}/')
                    for i in range(0, len(imgs)):
                        save_image(imgs[i]*0.5+0.5, f"./images/{folder}/img_{imgs_generated}_{kid_batch}.jpg")
                                                
                        # Saving the name of the image
                        imgs_path.append(f'img_{imgs_generated}_{kid_batch}.jpg')
                        
                        imgs_generated += 1
                        
                    # Reseting the variable if a batch its generated
                    epochs_no_generation = 0
                        
                else:
                    epochs_no_generation += 1
                       
                # If we have reached the number of images we want, we stop generating
                if imgs_generated >= num_imgs:
                    break
                    
    print(f'Data Augmentation finished. {imgs_generated} images have been generated.\n')
    
    ## Now we will create a dataset with the generated images
    # First we will create the dataframe with the images
    df = pd.DataFrame({'Diagnosis': 'Sick' if (sick_or_healthy == 1) else 'Healthy', 'Type': 'Static', 'Folder': f'{folder}', 'Image path': imgs_path})
    
    # Then we will create the dataset
    dataset = CancerDetectionDataset(f'./images', df)
    
    return dataset

def evaluate_images(imgs, val_dataset, device, step=5):
    """
    With this function we are computing KID_Score for the given images.
    """
    # Creating KID and FID variable
    KID = KernelInceptionDistance(subsets = 4, subset_size = 16, normalize = True).to(device)
    FID = FrechetInceptionDistance(normalize = True).to(device)
    
    # Updating validation real images first 
    val_loader = DataLoader(val_dataset, batch_size = 64)
    for img, _, _ in val_loader:
        img_val = (img.to(device)*0.5+0.5).repeat(1, 3, 1, 1)
        KID.update(img_val, real=True)
        FID.update(img_val, real=True)

    # Updating generated images
    KID.update((imgs*0.5+0.5).repeat(1, 3, 1, 1), real=False)
    FID.update((imgs*0.5+0.5).repeat(1, 3, 1, 1), real=False)        

    # Computing KID Score
    kid_score = float(KID.compute()[0].item())
    fid_score = float(FID.compute())
    
    # Resetting KID and FID variables
    KID.reset()
    FID.reset()

    return kid_score, fid_score


def compute_KID_FID_Score(gen, val_dataset, num_gen, device, step=5):
    """
    With this function we are computing KID_Score generating "num_gen" images with the generator provided. Also, we will compute the score by comparing with the validation dataset. 
    """
    # Creating KID and FID variable
    KID = KernelInceptionDistance(subsets = 4, subset_size = 5, normalize = True).to(device)
    FID = FrechetInceptionDistance(normalize = True).to(device)

    # Stopping gen evaluating to generate
    gen.eval()
    
    # Updating validation real images first 
    val_loader = DataLoader(val_dataset, batch_size = 64)
    for img, _, _ in val_loader:
        img_val = (img.to(device)*0.5+0.5).repeat(1, 3, 1, 1)
        KID.update(img_val, real=True)
        FID.update(img_val, real=True)

    # Updating KID_GEN_IMAGES generated images
    for i in range(0, round(num_gen/64)):
        noise = torch.randn(64, CONST.Z_DIM).to(device)
        img_gen = (gen(noise, 1.0, step)*0.5+0.5).repeat(1, 3, 1, 1)
        
        # If the model is StyleGAN, then we will set also the step for the generation
        KID.update(img_gen, real=False)
        FID.update(img_gen, real=False)
    
    # Computing KID Score
    kid_score = float(KID.compute()[0].item())
    fid_score = float(FID.compute())
    
    # Resetting KID and FID variables
    KID.reset()
    FID.reset()

    # Starting training again
    gen.train()
    
    return kid_score, fid_score
    

    
def split_sick_health(train_ids, dataset):
    """
    The purpose of this functions is to get the indexes of the sick and healty images and return the indexes of the subsets.
    """
    
    # We will append the indexes of the sick and healthy images
    # also we will append in a list all the rows of sick and healthy
    # into a dictionary variable
    sick_ids = []
    healthy_ids = []
    
    # Getting the IDs of the sick and healthy images
    for t_id in train_ids:
        if dataset.samples[t_id][1] == 0:
            healthy_ids.append(t_id)

        else:
            sick_ids.append(t_id)

    
    # Creating a new dataset for each one
    sick_dataset = dataset.subset(sick_ids)
    healthy_dataset = dataset.subset(healthy_ids)
    
    return sick_dataset, healthy_dataset


def split_train_val_gan(dataset):
    """
    This function will split the training between train and val.
    """
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
    y = [dataset.samples[i][1] for i in range(len(dataset.samples))]
    (train_ids, val_ids) = next(sgkf.split(dataset, y, dataset.groups))
    val_ids_to_remove = []
    for val_id in val_ids:
        if ("augmented" in dataset.samples[val_id][0].split("/")[-1]):
            index = np.where(val_ids==val_id)
            val_ids_to_remove.append(index)

    val_ids = np.delete(val_ids, val_ids_to_remove) 
    return train_ids, val_ids    


def save_samples(gen, device, train_string, step=0):
    """
    With this function we will save examples in each iteration of our training generator
    """
    for i in range(CONST.SAMPLES_GEN):
        with torch.no_grad():
            noise = torch.randn(1, CONST.Z_DIM).to(device)
            if 'StyleGAN' in str(gen):
                img = gen(noise, 1.0, step)
            else:
                img = gen(noise)
            
            if not os.path.exists(f'./images/{train_string}/step{step}'):
                os.makedirs(f'./images/{train_string}/step{step}')
            save_image(img*0.5+0.5, f"./images/{train_string}/step{step}/img_{i}.jpg")


def plot_loss(gen_loss, dis_loss, file_name = 'sGan graph.png'):
    """
    We will plot the loss function graph for generator and discriminator.
    
    Params:
        gen_loss: list of the function loss for the generator
        dis_loss: list of the function loss for the discriminator
        kid_score: list of the KID Score obtained through the training
        file_name: name of the image we will save
    """
    
    # Creating the figure of the graph
    fig, ax1 = plt.subplots()

    ## Loss
    # Adjusting the epochs
    epochs = np.arange(0, (len(CONST.BATCH_SIZES)*CONST.EPOCHS) + 1, CONST.SAMPLE_INTERVAL)
   
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    gen_line = ax1.plot(epochs, gen_loss, color = 'green', label='Generator')
    dis_line = ax1.plot(epochs, dis_loss, color = 'red', label='Discriminator')

    ## KID Score
    #ax2 = ax1.twinx()
    #
    #ax2.set_ylabel('KID Score')
    #ax2.yaxis.label.set_color('blue')
    #kid_line = ax2.plot(epochs, kid_score, color='blue', label='KID Score')
    
    # Putting the title and legend
    plt.title('Loss Function of Generator and Discriminator')
    
    lns = gen_line+dis_line#+kid_line
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Adding limits and grid
    ax1.grid()
    #ax1.set_ylim(0, 1.5)
    #ax2.set_ylim(0, 0.75)

    # Saving the image
    plt.savefig(file_name)
    
    
def plot_kid(kid_score, file_name = 'KID Score graph.png'):
    """
    We will plot the loss function graph for generator and discriminator.
    
    Params:
        kid_score: list of the KID Score obtained through the training
        file_name: name of the image we will save
    """
    
    # Creating the figure of the graph
    fig, ax1 = plt.subplots()

    ## Loss
    # Adjusting the epochs
    epochs = np.arange(0, (len(CONST.BATCH_SIZES)*CONST.EPOCHS) + 1, CONST.KID_SCORE_CHECKPOINT)
    
    #ax1.set_xlabel('Epochs')
    #ax1.set_ylabel('Loss')
    #gen_line = ax1.plot(epochs, gen_loss, color = 'green', label='Generator')
    #dis_line = ax1.plot(epochs, dis_loss, color = 'red', label='Discriminator')

    ## KID Score
    #ax2 = ax1.twinx()
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('KID')
    #ax1.yaxis.label.set_color('blue')
    kid_line = ax1.plot(epochs, kid_score, color='blue', label='KID Score')
    
    # Putting the title and legend
    plt.title('KID Score of the generated images')
    
    lns = kid_line#gen_line+dis_line#+kid_line
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Adding limits and grid
    ax1.grid()
    #ax1.set_ylim(0, 1.5)
    #ax2.set_ylim(0, 0.75)

    # Saving the image
    plt.savefig(file_name)


def plot_fid(fid_score, file_name = 'KID Score graph.png'):
    """
    We will plot the loss function graph for generator and discriminator.
    
    Params:
        kid_score: list of the KID Score obtained through the training
        file_name: name of the image we will save
    """
    
    # Creating the figure of the graph
    fig, ax1 = plt.subplots()

    ## Loss
    # Adjusting the epochs
    epochs = np.arange(0, (len(CONST.BATCH_SIZES)*CONST.EPOCHS) + 1, CONST.KID_SCORE_CHECKPOINT)
    
    #ax1.set_xlabel('Epochs')
    #ax1.set_ylabel('Loss')
    #gen_line = ax1.plot(epochs, gen_loss, color = 'green', label='Generator')
    #dis_line = ax1.plot(epochs, dis_loss, color = 'red', label='Discriminator')

    ## KID Score
    #ax2 = ax1.twinx()
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('FID')
    #ax1.yaxis.label.set_color('blue')
    fid_line = ax1.plot(epochs, fid_score, color='blue', label='FID Score')
    
    # Putting the title and legend
    plt.title('FID Score of the generated images')
    
    lns = fid_line#gen_line+dis_line#+kid_line
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Adding limits and grid
    ax1.grid()
    #ax1.set_ylim(0, 1.5)
    #ax2.set_ylim(0, 0.75)

    # Saving the image
    plt.savefig(file_name)