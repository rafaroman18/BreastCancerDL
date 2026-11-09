class CONSTANTS:
    EPOCHS = 60
    SAMPLE_INTERVAL = 30
    DATA_CHANNELS = 1
    NUM_CLASSES = 2
    
    # KID Score
    KID_SCORE_CHECKPOINT = 30
    KID_GEN_IMAGES = 1000
    
    # sGAN
    IMG_SIZE = 56
    LATENT_DIM = 100
    NUM_CLASSES_SGAN = 2
    BATCH_SIZE = 64
    FEATURE_MAP_GEN = 56
    FEATURE_MAP_DIS = 56
    LEARNING_RATE = 2e-4
    
    # Style GAN 
    LR = 1e-3
    LR_GEN = 1e-4
    B_UNO = 0.5
    B_DOS = 0.9
    
    FACTORS = [1,1,1,1/2,1/4,1/8,1/16,1/32]
    BATCH_SIZES =  [128, 128, 64, 64, 32, 32]
    STEP_IMG_SIZES =  [4, 8, 16, 32, 64, 128] 
    ALPHA_INIT = 1e-7
    Z_DIM = 512
    W_DIM = 512
    IN_CHANNELS = 512
    LAMBDA_GP = 10
    PROGRESSIVE_EPOCHS = [EPOCHS] * len(BATCH_SIZES)
    SAMPLES_GEN = 50
    
    # DATA AUGMENTATION
    DATA_AUGMENTATION_THRESHOLD = 0.25
    DATA_AUGMENTATION_NUM = 1152
    DATA_AUGMENTATION_SICK_FOLDER = 'DATA_AUGMENTATION_SICK/'
    DATA_AUGMENTATION_HEALTHY_FOLDER = 'DATA_AUGMENTATION_HEALTHY/'
    
    # BREAST CANCER DETECTION MODELS
    NUM_EPOCHS_CV = 50
    PATIENCE_CV = 10
    
    # BEST HYPERPARAMETERS
    SICK_HYPERPARAMETERS = [0.001, 0.0002, 0.5, 0.9, 0.25]
    HEALTHY_HYPERPARAMETERS = [0.001, 0.0001, 1e-05, 0.9, 0.25]
    
    # PATHS
    PATH_TRAIN_DATASET = './datasets/thermographic_breast_cancer_dataset_frontal_crop_images_augmented/'
    PATH_TRAIN_DATAFRAME = './datasets/frontal_crop_images_augmented.csv' 
    
    PATH_TEST_DATASET = './datasets/thermographic_breast_cancer_dataset_frontal_crop_images'
    PATH_TEST_DATAFRAME = './datasets/test_frontal_crop_images.csv' 


CONST = CONSTANTS()

