import pandas as pd
from models.styleGAN import *
from utils import save_samples, plot_kid, plot_fid, plot_loss, data_augmentation, compute_KID_FID_Score, not_leaking_data_augmentations
import time
from pathlib import Path
from datasets.cancer_detection_dataset import CancerDetectionDataset

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

def grad_penalty(dis, real, fake, alpha, train_step, device):
    
    # Getting the shape of the real batch
    BATCH_SIZE, C, H, W = real.shape
    
    # Interpolating the real images
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)
    
    # Calculating the critic output of the interpolated images
    mixed_scores = dis(interpolated_images, alpha, train_step)
    
    # Gradient
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gradient Penalty
    gradient = gradient.view(gradient.shape[0], -1)
    grad_penalty = torch.mean((gradient.norm(2, dim=1) - 1) ** 2)
    return grad_penalty


def training_function(dis, gen, dataset, loader, step, alpha, opt_dis, opt_gen, device):
        
    # Iterating through the real images in the loader
    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        cur_batch_size = imgs.shape[0]
        noise = torch.randn(cur_batch_size, CONST.Z_DIM).to(device)
        
        # Generating images
        fake = gen(noise, alpha, step)
        # fake = not_leaking_data_augmentations(gen(noise, alpha, step), 0.4)
        
        # Getting the critic output of real and generated images
        dis_real = dis(imgs, alpha, step)
        dis_fake = dis(fake.detach(), alpha, step)
        
        # Gradient Penalty 
        gp = grad_penalty(dis, imgs, fake, alpha, step, device)
        
        # Wassertein Loss
        dis_loss = (-(torch.mean(dis_real) - torch.mean(dis_fake))
                    + CONST.LAMBDA_GP * gp + (0.001) 
                    * torch.mean(dis_real ** 2))

        # Backpropagation
        dis.zero_grad()
        dis_loss.backward()
        opt_dis.step()

        gen_fake = dis(fake, alpha, step)
        gen_loss = -torch.mean(gen_fake)
        
        gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        
        # Calculating alpha (needed for the StyleGAN models)
        alpha += cur_batch_size / (
            CONST.PROGRESSIVE_EPOCHS[step] * 0.5 * len(dataset)
        )        
        alpha = min(alpha,1)

    return alpha, dis_loss.item(), gen_loss.item(), gp



def train_StyleGAN(device, train_dataset, val_dataset, test_dataset, train_string):#, patience=15):
    """
    In this function we will train and create our StyleGAN model with the images from the dataset.
    """
    total_epoch = 0
    
    # Creating the execution results folder
    Path(f'./training_results/{train_string}').mkdir(parents=True, exist_ok=True)
    Path(f'./images/{train_string}').mkdir(parents=True, exist_ok=True)
    
    # Training arrays
    gen_loss = []
    dis_loss = []
    
    # GAN Metrics
    kid_score = []   
    fid_score = []   

    # Creating the generator
    gen = Generator(CONST.Z_DIM, CONST.W_DIM, CONST.IN_CHANNELS, device).to(device)

    # Creatin the discriminator
    dis = Discriminator(CONST.IN_CHANNELS).to(device)

    # Optimizers
    ## Optimizer Gen
    opt_gen = optim.Adam([{'params': [param for name, param in gen.named_parameters() if 'map' not in name]},
                {'params': gen.map.parameters(), 'lr': CONST.LR_GEN}], lr=CONST.LR, betas =(CONST.B_UNO, CONST.B_DOS))

    ## Optimizer Dis
    opt_dis = optim.Adam(dis.parameters(), lr= CONST.LR, betas =(CONST.B_UNO, CONST.B_DOS))

    # Starting Training
    gen.train()
    dis.train()

    ## Training Loop
    step = 0
    gp = 0

    # Starting the timer
    start=time.time()

    ### Starting training ###
    for num_epochs in CONST.PROGRESSIVE_EPOCHS[step:]:        
        
        #best_kid_score = None
        
        print(f'===== STEP: {step} - IMAGE SIZE: {CONST.STEP_IMG_SIZES[step]} =====')
        
        alpha = CONST.ALPHA_INIT
        
        # Loading the dataset with the indicated size
        train_dataset.transform = transforms.Compose([
                                        transforms.Resize((CONST.STEP_IMG_SIZES[step],CONST.STEP_IMG_SIZES[step])),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                    ])
        train_loader = DataLoader(train_dataset, batch_size=CONST.BATCH_SIZES[int(log2(CONST.STEP_IMG_SIZES[step]/4))])
        
        for epoch in range(num_epochs):
            
            # Training function    
            alpha, d_loss, g_loss, gp = training_function(
                dis, gen, train_dataset, train_loader, step, alpha, opt_dis, opt_gen, device
            )
            
            # # Each KID_SCORE_CHECKPOINT we will save KID_Score
            if(total_epoch % CONST.KID_SCORE_CHECKPOINT == 0) | (total_epoch == ((len(CONST.BATCH_SIZES) * CONST.EPOCHS)-1)):

                kid__, fid__ = compute_KID_FID_Score(gen, val_dataset, CONST.KID_GEN_IMAGES, device, step)
                kid_score.append(kid__)
                fid_score.append(fid__)
                
            # Each SAMPLE_INTERVAL we will save the loss and prompt the loss and kid_score
            if(total_epoch % CONST.SAMPLE_INTERVAL == 0) | (total_epoch == ((len(CONST.BATCH_SIZES) * CONST.EPOCHS)-1)):
                
                # Saving Loss
                dis_loss.append(d_loss)
                gen_loss.append(g_loss)
                
                print(
                f"Epoch [{epoch + 1}/ {num_epochs}] [Image Size: {CONST.STEP_IMG_SIZES[step]}] [D loss: %f] [G loss: %f] [Last KID: %f] [Last FID: %f]"
                % (d_loss, g_loss, kid_score[-1], fid_score[-1])
                )
                
            total_epoch += 1
            
        # Saving samples
        # gen.eval()
        # save_samples(gen, device, train_string, step)
        # gen.train()
        
        step +=1
        
    # Ending the timer
    end = time.time()

    # Saving a file with the training info
    with open(f'./training_results/{train_string}/{train_string}.txt', "w") as text_file:
        text_file.write("LAST FUNCTION LOSS RESULTS: [Epoch %d/%d] [D loss: %f] [G loss: %f]\n\nLAST KID SCORE: %f\n\nLAST FID SCORE: %f\n\nTraining Time: %f minutes.\n" % (epoch, CONST.EPOCHS, d_loss, g_loss, kid_score[-1], fid_score[-1], (end-start)/60.0))
        text_file.write(f"\nKID_SCORE VALUES: {kid_score}\n")
        text_file.write(f"\nFID_SCORE VALUES: {fid_score}\n")
        
    print("Training Time: %f minutes." % ((end-start)/60.0))
    
    # Plotting loss and KID
    plot_loss(gen_loss, dis_loss, f'./training_results/{train_string}/Loss_{train_string}.png')  
    plot_kid(kid_score, f'./training_results/{train_string}/KIDScore_{train_string}.png')
    plot_fid(fid_score, f'./training_results/{train_string}/FIDScore_{train_string}.png')

    # Setting the models to evaluation before returning them
    gen.eval()
    dis.eval()
    
    return gen, dis

########################################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       

# # Dataframes
# df_train = pd.read_csv(CONST.PATH_TRAIN_DATAFRAME)
# df_validation = df_train[1980:]
# df_train = df_train[:1980]

# df_test = pd.read_csv(CONST.PATH_TEST_DATAFRAME)

# print(f'Longitud de TRAIN: {len(df_train)}')
# print(f'Longitud de VAL: {len(df_validation)}')

# # Train and validation
# dataset_train = CancerDetectionDataset(CONST.PATH_TRAIN_DATASET, df_train)
# dataset_validation = CancerDetectionDataset(CONST.PATH_TRAIN_DATASET, df_validation)
# dataset_test = CancerDetectionDataset(CONST.PATH_TEST_DATASET, df_test)

# train_StyleGAN(device, dataset_train, dataset_validation, dataset_test, f'TESTING HYPERPARAMETERS WITH NON LEAKING FILTER - {CONST.EPOCHS} EPOCHS')
