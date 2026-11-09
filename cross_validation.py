import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from utils import split_sick_health, data_augmentation, split_train_val_gan
from models.styleGAN import Generator, Discriminator
from models.breast_cancer_detection_net import BreastCancerDetectionNet
from models.breast_cancer_detection_self_attention_net import BreastCancerDetectionSelfAttentionNet
from models.breast_cancer_detection_soft_attention_net import BreastCancerDetectionSoftAttentionNet
from models.breast_cancer_detection_hard_attention_net import BreastCancerDetectionHardAttentionNet
from datasets.cancer_detection_dataset import CancerDetectionDataset
from constants import CONST
import torchvision
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from train_StyleGAN import train_StyleGAN
import time
import sklearn.metrics as metrics
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

###### CROSS-VALIDATION                      PATH_TRAIN_DATAFRAME, PATH_TRAIN_DATASET, DATA_AUGMENTATION_NUM
def cross_validation(model_class, path_dataframe_images, path_dataset_images, num_data_aug, training_string, num_folds = 5, num_epochs = 1, patience = 10):

    print(f'\nModel: {str(model_class())}')
    print(f'\nGAN: StyleGAN')
    print(f'Number of folds: {num_folds}')
    print(f'Number of epochs: {num_epochs}')

    df_train = pd.read_csv(path_dataframe_images)
    dataset = CancerDetectionDataset(path_dataset_images, df_train)
    
    validation_scores = []    
    accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    precision_scores = []
    f1_scores = []
    val_labels_list = []
    val_preds_list = []
    save_model = True

    ### 1ST STEP
    #  DIVIDE THE DATA BETWEEN TRAINING, VAL AND TEST FOR THIS EPOCH
    start = time.time()
    sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True)
    y = [dataset.samples[i][1] for i in range(len(dataset.samples))]
    for fold, (train_ids, test_ids) in enumerate(sgkf.split(dataset, y, dataset.groups)):
        print('\nFold [{}/{}]...\n'.format(fold+1, num_folds))

        test_ids_to_remove = []
        for test_id in test_ids:
            if ("augmented" in dataset.samples[test_id][0].split("/")[-1]):
                index = np.where(test_ids==test_id)
                test_ids_to_remove.append(index)

        # Removing the augmented images from test
        test_ids = np.delete(test_ids, test_ids_to_remove) 
        test_dataset = dataset.subset(test_ids)

        # Creating the dataset with the train ids (we will concatenate the augmented and this dataset)
        original_train_dataset = dataset.subset(train_ids)
        print(f'LONGITUD DE TRAIN ORIGINAL: {len(original_train_dataset)}')

        # We now divide the training set between train and val (val will be used for KID Score and FID Score)
        train_gan_ids, val_ids = split_train_val_gan(dataset.subset(train_ids))
        val_dataset = dataset.subset(val_ids)
        
        # Getting the dataset splits of the sick and healthy images from the current train split
        sick_dataset, healthy_dataset = split_sick_health(train_gan_ids, dataset)

        # print(f'DATASET LEN: {len(dataset)}')
        # print(f'TRAIN - LONGITUD: {len(train_gan_ids)}')
        # print(f'ORIGINAL TRAIN - LONGITUD: {len(original_train_dataset)}')
        # print(f'VAL - LONGITUD: {len(val_ids)}')
        # print(f'TEST - LONGITUD: {len(test_dataset)}')
        # print(f'SICK - LONGITUD: {len(sick_dataset)}')
        # print(f'HEALTHY - LONGITUD: {len(healthy_dataset)}')
                
    ### 2ND STEP
    # TRAIN STYLE_GAN 
        # StyleGAN for sick images
        print('--- Training with Sick Images ---\n')
        # Setting the hyperparameters for sick model
        CONST.LR, CONST.LR_GEN, CONST.B_UNO, CONST.B_DOS, _ = CONST.SICK_HYPERPARAMETERS
        gen_sick, _ = train_StyleGAN(device, sick_dataset, val_dataset, test_dataset, f'Sick - {training_string} - Fold {fold}')
        
        # StyleGAN for healthy images
        print('\n--- Training with Healthy Images ---\n')
        # Setting the hyperparameters for healthy model
        CONST.LR, CONST.LR_GEN, CONST.B_UNO, CONST.B_DOS, _ = CONST.HEALTHY_HYPERPARAMETERS
        gen_healthy, _ = train_StyleGAN(device, healthy_dataset, val_dataset, test_dataset, f'Healthy - {training_string} - Fold {fold}')
    
    ### 3RD STEP 
    # GENERATE IMAGES (THE BEST ONES BY KID SCORE) AND CONCATENATE THE TRAINING AND THE NEW IMAGES
        print()
        start_DA = time.time()
        # Data Augmentation for Sick Images
        #CONST.DATA_AUGMENTATION_THRESHOLD = CONST.SICK_HYPERPARAMETERS[-1]
        sick_data_aug = data_augmentation(gen_sick, val_dataset, CONST.DATA_AUGMENTATION_NUM, 1, device, training_string + f' - Fold {fold}')
        
        # Data Augmentation for Healthy Images
        #CONST.DATA_AUGMENTATION_THRESHOLD = CONST.HEALTHY_HYPERPARAMETERS[-1]
        healthy_data_aug = data_augmentation(gen_healthy, val_dataset, CONST.DATA_AUGMENTATION_NUM, 0, device, training_string + f' - Fold {fold}')
        end_DA = time.time()
        print("Data Augmentation: %f minutes." % ((end_DA-start_DA)/60.0))
        
        # Transform all the datasets into 56x56 images      
        new_transform = transforms.Compose([transforms.Resize((CONST.IMG_SIZE ,CONST.IMG_SIZE)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.5,), (0.5,))])

        # The train images will have 56x56 size
        original_train_dataset.transform = new_transform
        sick_data_aug.transform = new_transform
        healthy_data_aug.transform = new_transform
        
        # Transform and generate the test dataset 
        test_dataset.transform = new_transform
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
        # Concatenate all the datasets
        dataset_def = ConcatDataset([original_train_dataset, sick_data_aug, healthy_data_aug])
        print(f'LONGITUD DEL DATASET: {len(dataset_def)}')
        train_loader = DataLoader(dataset_def, batch_size=64, shuffle=True)

        # print(f'DATASET_DEF - LONGITUD: {len(dataset_def)}')
        
        print(f'\n--- Training Model: {str(model_class())} ---')
        
    ### 4TH STEP
    # TRAIN JOSE CARLOS' MODEL
        model = model_class()
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_loss = None
        best_model = model
        no_improvement_count = 0

        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0

            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            
            # Calculate validation loss
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(test_loader)
            model.train()
            
            print(f' · Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print('\n · No improvement in validation loss for {} epochs, stopping...'.format(patience))
                break

        print()
    # # ### 5TH STEP
    # # # GET RESULTS AND ANALYSE
        model = best_model
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            correct, total = 0, 0
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

            validation_score = 100 * correct / total
            validation_scores.append(validation_score)

        # Saving the model
        if save_model:
            # if not os.path.exists(f'./trained_models/{training_string}'):
            #     os.makedirs(f'./trained_models/{training_string}')
            torch.save(model.state_dict(), f'./trained_models/{training_string}.pth')
            save_model = False

        accuracy_scores.append(accuracy_score(val_labels, val_preds))
        sensitivity_scores.append(recall_score(val_labels, val_preds))
        precision_scores.append(precision_score(val_labels, val_preds, zero_division=0))
        f1_scores.append(f1_score(val_labels, val_preds))
        val_labels_list.extend(val_labels)
        val_preds_list.extend(val_preds)

        cm = confusion_matrix(val_labels, val_preds)
        
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity_scores.append(specificity)
        
        print(' · Validation accuracy: %.2f%%' % (validation_score))
        print(f' · Confusion Matrix:   {cm[0]}')
        print(f'{" ":<23}{cm[1]}')
        
        # Before the next split we will put dataset into the old transform again
        dataset.transform = transforms.Compose([
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])
    end = time.time()
    print('\n\nEXACTITUD - Average validation accuracy: %.2f%%' % (sum(validation_scores) / len(validation_scores)))
    print('ESPECIFICIDAD - Average specificity: %.2f%%' % (100 * sum(specificity_scores) / len(specificity_scores)))    
    print('PRECISIÓN - Average precision: %.2f%%' % (100 * sum(precision_scores) / len(precision_scores)))
    print('EXHAUSTIVIDAD - Average sensitivity: %.2f%%' % (100 * sum(sensitivity_scores) / len(sensitivity_scores)))
    print('F1 SCORE - Average F1-score: %.2f%%\n' % (100 * sum(f1_scores) / len(f1_scores)))
    print()
    print("CROSS VALIDATION Time: %f minutes." % ((end-start)/60.0))
    
    return {
        'Exactitud': validation_scores,
        'Especificidad':  specificity_scores,
        'Precision': precision_scores,
        'Exhaustividad': sensitivity_scores,
        'F1_Score': f1_scores,
        'Val_Labels': val_labels_list,
        'Val_Preds': val_preds_list
    }

training_string = f'Nuevas ejecuciones_HARD_CV - 2304 GEN - {CONST.EPOCHS} epochs'

exactitud = []
especificidad = []
precision = []
exhaustividad = []
f1_score_ = []
val_labels = []
val_preds = []

for i in range(1, 11):
    
    print(f'===================================================================================')
    print(f'======================== SEED RUN {i} =========================')
    print(f'===================================================================================')

    results = cross_validation(BreastCancerDetectionHardAttentionNet, CONST.PATH_TRAIN_DATAFRAME, CONST.PATH_TRAIN_DATASET, CONST.DATA_AUGMENTATION_NUM, f'{training_string}_RUN_{i}', num_folds = 5, num_epochs = CONST.NUM_EPOCHS_CV, patience = CONST.PATIENCE_CV)

    exactitud.append(results['Exactitud'])
    especificidad.append(results['Especificidad'])
    precision.append(results['Precision'])
    exhaustividad.append(results['Exhaustividad'])
    f1_score_.append(results['F1_Score'])
    val_labels.extend(results['Val_Labels'])
    val_preds.extend(results['Val_Preds'])
    
    print()
    print()

print(f'===================================================================================')
print(f'======================== FINAL RESULTS =========================')
print(f'===================================================================================')

print(f'EXACTITUD - MEAN {np.mean(exactitud)} - VARIANCE {np.var(exactitud)} - STANDAR DEVIATION {np.std(exactitud)}')
print(f'ESPECIFICIDAD - MEAN {np.mean(especificidad)} - VARIANCE {np.var(especificidad)} - STANDAR DEVIATION {np.std(especificidad)}')
print(f'PRECISION - MEAN {np.mean(precision)} - VARIANCE {np.var(precision)} - STANDAR DEVIATION {np.std(precision)}')
print(f'EXHAUSTIVIDAD - MEAN {np.mean(exhaustividad)} - VARIANCE {np.var(exhaustividad)} - STANDAR DEVIATION {np.std(exhaustividad)}')
print(f'F1_SCORE - MEAN {np.mean(f1_score_)} - VARIANCE {np.var(f1_score_)} - STANDAR DEVIATION {np.std(f1_score_)}')

if not os.path.exists(f'./training_results/{training_string}'):
    os.makedirs(f'./training_results/{training_string}')

# '------------------ CONFUSION_MATRIX -------------------------'
cm = confusion_matrix(val_labels, val_preds)
ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)
plt.savefig(f'./training_results/{training_string}/CONFUSION_MATRIX_{training_string}.png')
plt.close()

# '------------------ AUC ROC CURVE -------------------------'
fpr, tpr, thresholds = metrics.roc_curve(val_labels, val_preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Curva ROC y valor AUC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Ratio de Verdaderos Positivos')
plt.xlabel('Ratio de Falsos Positivos')
plt.savefig(f'./training_results/{training_string}/CURVA_ROC_AUC_{training_string}.png')
