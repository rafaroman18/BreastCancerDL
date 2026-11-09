import torch
import pandas as pd
from constants import CONST# import LR, PATH_TRAIN_DATAFRAME, PATH_TRAIN_DATASET, BATCH_SIZES
from datasets.cancer_detection_dataset import CancerDetectionDataset
from utils import compute_KID_FID_Score, split_sick_health
import time
from train_StyleGAN import train_StyleGAN
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

# GridSearch Possible Values
LEARNING_RATE_DIS_GS = [1e-3, 2e-4, 1e-4]
LEARNING_RATE_GEN_GS = [1e-5, 2e-4, 1e-4]
B_UNO_GS = [1e-05, 0.5]
B_DOS_GS = [0.9, 0.999]
EPOCHS_GS = [60]

print('--------------------------------------------')
print('Starting hyperparameters tuning. GridSearch.')
print('Values:\n')
print(f'LR: {LEARNING_RATE_DIS_GS}')
print(f'LR_GEN: {LEARNING_RATE_GEN_GS}')
print(f'B_UNO: {B_UNO_GS}')
print(f'B_DOS: {B_DOS_GS}')
print(f'EPOCHS: {EPOCHS_GS}')
print('--------------------------------------------')
print()

# Best results
every_result = []
best_result = {}
best_kid_score = 10000


# Data
df_train = pd.read_csv(CONST.PATH_TRAIN_DATAFRAME)
dataset = CancerDetectionDataset(CONST.PATH_TRAIN_DATASET, df_train)

# Selecting onlyt sick or healthy images
_, dataset = split_sick_health(list(range(0, len(dataset))), dataset)

print(f'LONGITUD DE LAS IMAGENES SANAS: {len(dataset)}')

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
val_dataset = dataset.subset(validation_ids)
train_dataset = dataset.subset(train_ids)

print(f'TRAIN - LONGITUD DE LAS IMAGENES SANAS: {len(train_dataset)}')
print(f'TEST - LONGITUD DE LAS IMAGENES SANAS: {len(val_dataset)}')

# Startig timer
start = time.time()

# Training
for lr_dis in LEARNING_RATE_DIS_GS:
    for lr_gen in LEARNING_RATE_GEN_GS:
        for b_1 in B_UNO_GS:
            for b_2 in B_DOS_GS:
                for ep in EPOCHS_GS:
                    # if  not(
                    #     ((lr_dis == 0.001) & (lr_gen == 1e-05) & (b_1 == 0) & (b_2 == 0.9)) |
                    #     ((lr_dis == 0.001) & (lr_gen == 1e-05) & (b_1 == 0) & (b_2 == 0.999)) |
                    #     ((lr_dis == 0.001) & (lr_gen == 1e-05) & (b_1 == 0.5) & (b_2 == 0.9)) |
                    #     ((lr_dis == 0.001) & (lr_gen == 1e-05) & (b_1 == 0.5) & (b_2 == 0.999)) |
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0002) & (b_1 == 0) & (b_2 == 0.9)) |
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0002) & (b_1 == 0) & (b_2 == 0.999)) | 
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0002) & (b_1 == 0.5) & (b_2 == 0.9)) |                         
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0002) & (b_1 == 0.5) & (b_2 == 0.999)) |                         
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0001) & (b_1 == 0) & (b_2 == 0.9)) |  
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0001) & (b_1 == 0) & (b_2 == 0.999)) |                         
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0001) & (b_1 == 0.5) & (b_2 == 0.9)) |                         
                    #     ((lr_dis == 0.001) & (lr_gen == 0.0001) & (b_1 == 0.5) & (b_2 == 0.999)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 1e-05) & (b_1 == 0) & (b_2 == 0.9)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 1e-05) & (b_1 == 0) & (b_2 == 0.999)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 1e-05) & (b_1 == 0.5) & (b_2 == 0.9)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 1e-05) & (b_1 == 0.5) & (b_2 == 0.999)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 0.0002) & (b_1 == 0) & (b_2 == 0.9)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 0.0002) & (b_1 == 0) & (b_2 == 0.999)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 0.0002) & (b_1 == 0.5) & (b_2 == 0.9)) |                         
                    #     ((lr_dis == 0.0002) & (lr_gen == 0.0002) & (b_1 == 0.5) & (b_2 == 0.999)) |
                    #     ((lr_dis == 0.0002) & (lr_gen == 0.0001) & (b_1 == 0) & (b_2 == 0.9)) |
                    #     ((lr_dis == 0.0002) & (lr_gen == 0.0001) & (b_1 == 0) & (b_2 == 0.999))
                    # ): 
                            
                            # New Values of gridsearch
                            CONST.LR = lr_dis
                            CONST.LR_GEN = lr_gen
                            CONST.B_UNO = b_1
                            CONST.B_DOS = b_2
                            CONST.EPOCHS = ep
                            CONST.PROGRESSIVE_EPOCHS = [ep] * len(CONST.BATCH_SIZES)
                            print(f'Current Values: {CONST.LR} LEARNING RATE - {CONST.LR_GEN} LEARNING RATE GEN - {CONST.B_UNO} B_1 - {CONST.B_DOS} B_2 - {CONST.EPOCHS} epochs')
                            print()
            
                            # Training
                            gen, dis = train_StyleGAN(device, train_dataset, val_dataset, val_dataset, f'HEALTHY - HyperTuning - {CONST.LR} LR DIS - {CONST.LR_GEN} LR GEN - {CONST.B_UNO} B_1 - {CONST.B_DOS} B_2 - {CONST.EPOCHS} epochs')
                    
                            # Evaluation
                            cur_kid_score, cur_fid_score = compute_KID_FID_Score(gen, val_dataset, 10000, device)
            
                            print()
                            print(f'----- Current KID Score: {cur_kid_score} || Current FID Score: {cur_fid_score} -----')
                            print()
                            every_result.append({'KID Score': cur_kid_score, 'FID Score': cur_fid_score, 'LR DIS': CONST.LR, 'LR GEN': CONST.LR_GEN, 'B_UNO': CONST.B_UNO, 'B_DOS': CONST.B_DOS, 'EPOCHS': CONST.EPOCHS})
                            
                            if cur_kid_score < best_kid_score:
                                best_kid_score = cur_kid_score
                                best_result = {'KID Score': cur_kid_score, 'FID Score': cur_fid_score, 'LR DIS': CONST.LR, 'LR GEN': CONST.LR_GEN, 'B_UNO': CONST.B_UNO, 'B_DOS': CONST.B_DOS, 'EPOCHS': CONST.EPOCHS}

# Stopping timer
end = time.time()

# Result
print(f'The best result is LEARNING RATE: {best_result['LR DIS']} - LEARNING RATE GEN: {best_result['LR GEN']} - B_1: {best_result['B_UNO']} - B_2: {best_result['B_DOS']} - EPOCHS: {best_result['EPOCHS']} with KID Score {best_result['KID Score']} and FID Score {best_result['FID Score']}.')

print()

print("Hyperparameter Tuning Time: %f minutes." % ((end-start)/60.0))

print()
print('And all the results are:')
for i in every_result:
    print(i)
