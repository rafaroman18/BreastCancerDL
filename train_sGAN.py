from sGAN import *
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import save_samples
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import time
from utils import plot_loss, plot_kid, compute_KID_FID_Score, split_sick_health
from torchmetrics.image.kid import KernelInceptionDistance
from pathlib import Path
import warnings
from constants import CONST
from sklearn.model_selection import StratifiedGroupKFold


# KID Score always throw a Warning to inform about memory usage. We are going to ignore it
warnings.filterwarnings("ignore")

def train_sgan(device, train_dataset, val_dataset, test_dataset, training_string):
    """
    In this function we will train and create our sGAN model with the images
    from the dataset.
    """

    # Creating the execution results folder
    Path(f'./training_results/{training_string}').mkdir(parents=True, exist_ok=True)
    Path(f'./images/{training_string}').mkdir(parents=True, exist_ok=True)
    
    print('Starting the training of sGAN')
    print(f'Number of epochs: {CONST.EPOCHS}')
    
    # We will create the list of function loss for generator and discriminator
    gen_loss = []
    dis_loss = []
    
    # Also, we will create the KID Score variable and the lists of each epoch obtained
    KID = KernelInceptionDistance(subsets = 64, subset_size = 20, normalize = True).to(device)
    kid_score = []

    gen = Generator().to(device)      
    dis = Discriminator().to(device)
    
    # Loss functions
    adversarial_loss = nn.BCELoss().to(device)

    ##########################################
    ######## !!!!!!!! AQUI HABRIA QUE PONER BCELOSS PORQUE EL MODELO SOLO VA A TENER QUE DISTINGUIR ENTRE FALSO Y REAL (DENTRO DE REAL SOLO HABRA SANO O ENFERMO PORQUE VAMOS A PONER DOS MODELOS) ¡¡¡¡¡¡¡¡
    classifier_loss = nn.CrossEntropyLoss().to(device)
    ##########################################
    
    # Initialize weights
    gen.apply(weights_init_normal)
    dis.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=CONST.LEARNING_RATE, 
                                   betas=(CONST.B_UNO, CONST.B_DOS))
    optimizer_D = torch.optim.Adam(dis.parameters(), lr=CONST.LEARNING_RATE, 
                                   betas=(CONST.B_UNO, CONST.B_DOS))   
    
    FloatTensor = torch.cuda.FloatTensor if device else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device else torch.LongTensor

    # Creating the data loader
    train_loader = DataLoader(train_dataset, batch_size=CONST.BATCH_SIZE, shuffle=True)

    # Starting timer
    start=time.time()

    ### Starting training ###
    for epoch in range(CONST.EPOCHS):
        for i, (inputs, labels, _) in enumerate(train_loader):
            
            actual_batch_size = inputs.shape[0]
        
            # Adversarial ground truths
            valid = FloatTensor(actual_batch_size, 1).fill_(1.0).to(device)
            fake = FloatTensor(actual_batch_size, 1).fill_(0.0).to(device)
            
            fake_aux_gt = LongTensor(actual_batch_size).fill_(2).to(device)

            # Configure input
            real_imgs = inputs.type(FloatTensor).to(device)
            labels = labels.type(LongTensor).to(device)

            ### Training GEN ###
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = FloatTensor(torch.randn(actual_batch_size, 3, 7, 7, device=device)).to(device)

            # Generate a batch of images
            gen_imgs = gen(z) 

            # Loss measures generator's ability to fool the discriminator
            generated, _ = dis(gen_imgs)
            g_loss = adversarial_loss(generated, valid)            
            
            g_loss.backward()
            optimizer_G.step()
            
            ### Training DIS ###            
            optimizer_D.zero_grad()
            print(real_imgs.shape)
            print(gen_imgs.shape)
            # Loss for real images
            real_generated, real_class = dis(real_imgs)
            d_real_loss = (adversarial_loss(real_generated, valid) + classifier_loss(real_class, labels)) / 2.0
        
            # Loss for fake images
            fake_generated, fake_class = dis(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_generated, fake) + classifier_loss(fake_class, fake_aux_gt)) / 2.0

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2.0
        
            # Calculate discriminator accuracy
            pred = np.concatenate([real_class.data.cpu().numpy(), fake_class.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimizer_D.step()

        # Each CONST.KID_SCORE_CHECKPOINT we will save KID_Score
        if(epoch % CONST.KID_SCORE_CHECKPOINT == 0) | (epoch == (EPOCHS-1)):
            kid_score.append(compute_KID_Score(KID, gen, val_dataset, KID_GEN_IMAGES, device))
            
        # Each CONST.SAMPLE_INTERVAL epochs we will....
        if (epoch % CONST.SAMPLE_INTERVAL == 0) | (epoch == (EPOCHS-1)):

            # ===========================================================================
            # print(f'G_LOSS: {g_loss.item()}')
            # print(f'D_LOSS: {d_loss.item()}')
            # print(f'D_REAL_LOSS: {d_real_loss}')
            # print(f'D_FAKE_LOSS: {d_fake_loss}')
            # print(f'D_ADVERSARIAL_LOSS: {adversarial_loss(fake_generated, fake)}') 
            # print(f'D_CLASSIFIER_LOSS: {classifier_loss(fake_class, fake_aux_gt)}') 
            # ===========================================================================    

            # Save the generator and discriminator loss
            gen_loss.append(g_loss.item())
            dis_loss.append(d_loss.item())

            # We will print gen/dis loss and KID Score
            print(
                "[Epoch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f] [Last KID Score Checkpoint: %f]"
                % (epoch + 1, EPOCHS, d_loss.item(), 100 * d_acc, g_loss.item(), kid_score[-1])
            )

            # We will save images each CONST.SAMPLE_INTERVAL epochs
            save_samples(gen, device, training_string, 0)
            # save_samples(gen, 0(training_string, epoch,training_string), nrow=1, normalize=True)

    end = time.time()
    
    # After the training, we will save a graph with function loss, KID score and a txt with some training data 
    plot_loss(gen_loss, dis_loss, f'./training_results/{training_string}/Loss_{training_string}.png')    
    plot_kid(kid_score, f'./training_results/{training_string}/KidScore_{training_string}.png')

    with open(f'./training_results/{training_string}/{training_string}.txt', "w") as text_file:
        text_file.write("LAST FUNCTION LOSS RESULTS: [Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]\n\nLAST KID SCORE: %f\n\nTraining Time: %f minutes." % (epoch, EPOCHS, i, len(train_loader), d_loss.item(), 100 * d_acc, g_loss.item(), kid_score[-1], (end-start)/60.0))
    print("Training Time: %f minutes." % ((end-start)/60.0))

    # We will save the trained model
    torch.save(gen.state_dict(), f'./trained_models/GEN - {training_string}.pt')
    torch.save(dis.state_dict(), f'./trained_models/DIS - {training_string}.pt')
    print(f'Trained models have been saved in the folder /trained_models/') 

    # Finally we return the loss, accuracy and KID Score
    return [d_loss.item(), g_loss.item(), 100 * d_acc, kid_score[-1]]

#=================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

# Datasets
df_train = pd.read_csv(CONST.PATH_TRAIN_DATAFRAME)
dataset = CancerDetectionDataset(CONST.PATH_TRAIN_DATASET, df_train)

# Inside sick or healthy images, splitting between train and test
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
y = [dataset.samples[i][1] for i in range(len(dataset.samples))]
train_ids, validation_ids = next(sgkf.split(dataset, y, dataset.groups))

val_ids_to_remove = []
for val_id in validation_ids:
    if ("augmented" in dataset.samples[val_id][0].split("/")[-1]):
        index = np.where(validation_ids==val_id)
        val_ids_to_remove.append(index)

validation_ids = np.delete(validation_ids, val_ids_to_remove) 

# Train/Test
new_transform = transforms.Compose([transforms.Resize((CONST.IMG_SIZE ,CONST.IMG_SIZE)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.5,), (0.5,))])

val_dataset = dataset.subset(validation_ids)
train_dataset = dataset.subset(train_ids)

val_dataset.transform = new_transform
train_dataset.transform = new_transform

# Training the models
train_sgan(device, train_dataset, val_dataset, val_dataset, f'SGan - Obteniendo imagenes para la memoria - {CONST.EPOCHS} epochs')