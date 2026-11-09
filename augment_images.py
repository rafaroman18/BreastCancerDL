import os
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

PATH_DATASET_IMAGES = "./datasets/thermographic_breast_cancer_dataset_frontal_crop_images"
PATH_DATASET_AUGMENTED_IMAGES = "./datasets/thermographic_breast_cancer_dataset_frontal_crop_images_augmented"
PATH_DATAFRAME_AUGMENTED_IMAGES = "./datasets/test_frontal_crop_images_augmented.csv"
NUM_IMAGES = 10

transform_only_horizontal_flip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1)
])

transform = transforms.Compose([
    #transforms.RandomRotation(degrees=1),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

if __name__ == "__main__":

    if not os.path.exists(PATH_DATASET_AUGMENTED_IMAGES):
        os.mkdir(PATH_DATASET_AUGMENTED_IMAGES)
    
    df_crop_images = pd.read_csv('./datasets/test_frontal_crop_images.csv')
    data = []

    for index, row in df_crop_images.iterrows():
        folder = row['Folder']
        filename = row['Image path']
        diagnosis = row['Diagnosis']
        type = row['Type']

        if not os.path.exists(f"{PATH_DATASET_AUGMENTED_IMAGES}/{folder}"):
            os.mkdir(f"{PATH_DATASET_AUGMENTED_IMAGES}/{folder}")

        image_path = f"{PATH_DATASET_IMAGES}/{folder}/{filename}"
        image = Image.open(image_path)

        image.save(f"{PATH_DATASET_AUGMENTED_IMAGES}/{folder}/{filename}")

        data.append([diagnosis, type, folder, filename])

        for i in range(NUM_IMAGES):

            if (i == 0):
                augmented_image = transform_only_horizontal_flip(image)
            else:
                augmented_image = transform(image)
                
            augmented_image_filename = f"{filename[:-4]}_augmented_{i+1}.jpg"
            augmented_image.save(f"{PATH_DATASET_AUGMENTED_IMAGES}/{row['Folder']}/{augmented_image_filename}")
            data.append([diagnosis, type, folder, augmented_image_filename])

    df_crop_images_augmented = pd.DataFrame(data, columns=['Diagnosis', 'Type', 'Folder', 'Image path'])
    df_crop_images_augmented.to_csv(PATH_DATAFRAME_AUGMENTED_IMAGES, index=False)