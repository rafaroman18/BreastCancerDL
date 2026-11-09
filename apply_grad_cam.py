from models.breast_cancer_detection_hard_attention_net import BreastCancerDetectionHardAttentionNet
from datasets.cancer_detection_dataset import CancerDetectionDataset
from constants import CONST
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision

def apply_grad_cam(model_class, path_dataframe_images, path_dataset_images, device, path_model):

    print(f'\nApplying Grad Cam to model: {str(model_class())}\n')

    model = model_class()
    model.load_state_dict(torch.load(path_model))
    model.to(device)
    model.requires_grad_(True)
    model.train()

    path_grad_cam = "./results/grad_cam"

    if not os.path.exists(path_grad_cam):
        os.mkdir(path_grad_cam)

    if not os.path.exists(f"{path_grad_cam}/{str(model_class())}"):
        os.mkdir(f"{path_grad_cam}/{str(model_class())}")

    def forward_hook(module,input, output):
        activation.append(output)

    def backward_hook(module, grad_in, grad_out):
        grad.append(grad_out[0])

    model.cnn[-1].register_forward_hook(forward_hook)
    model.cnn[-1].register_full_backward_hook(backward_hook)

    df_test = pd.read_csv(path_dataframe_images)
    test_dataset = CancerDetectionDataset(path_dataset_images, df_test)

        # Searching for patient 192
    id_ = 0
    for i in range(0,len(test_dataset)):
        if ("T0192.1.1.S.2013-09-06.00.jpg" in test_dataset[i][2]):
            print(f'ENCONTRADO: {test_dataset[i][2]}')
            id_ = i
            break

    test_dataset = test_dataset.subset([id_])
    
    test_dataset.transform = transforms.Compose([transforms.Resize((CONST.IMG_SIZE ,CONST.IMG_SIZE)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.5,), (0.5,))])


    test_loader = DataLoader(test_dataset, batch_size=1)

    for input, label, image_path in test_loader:

        grad=[]
        activation=[]

        input.requires_grad_(True)
        input, label = input.to(device), label.to(device)
        out=model(input)

        loss=out[0, label.item()]

        model.zero_grad()

        loss.backward()

        grads=grad[0].cpu().data.numpy().squeeze()
        fmap=activation[0].cpu().data.numpy().squeeze()

        tmp=grads.reshape([grads.shape[0],-1])
        weights=np.mean(tmp,axis=1)

        cam = np.zeros(grads.shape[1:])

        for i,w in enumerate(weights):
            cam += w*fmap[i,:]

        # cam = cam / cam.max() * 255
        cam=(cam>0)*cam
        max_value = cam.max()
        if max_value != 0:
            cam = cam / max_value * 255

        img = Image.open(image_path[0]).convert('L')

        npic=np.array(img)

        cam = cv2.resize(cam, (npic.shape[1], npic.shape[0]))

        heatmap=cv2.applyColorMap(np.uint8(cam),cv2.COLORMAP_JET)

        npic_color = cv2.cvtColor(npic, cv2.COLOR_GRAY2BGR)

        cam_img=npic_color*0.7+heatmap*0.3

        # cam_img=npic*0.7+heatmap*0.3

        plt.imshow(torchvision.transforms.ToPILImage()(np.uint8(cam_img[:,:,::-1])))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{path_grad_cam}/{str(model_class())}/{model_class()}-{image_path[0].split('/')[-1]}", bbox_inches='tight')
        # plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

apply_grad_cam(BreastCancerDetectionHardAttentionNet, CONST.PATH_TRAIN_DATAFRAME, CONST.PATH_TRAIN_DATASET, device, './trained_models/Nuevas ejecuciones_HARD_CV - 2304 GEN - 60 epochs_RUN_8.pth')