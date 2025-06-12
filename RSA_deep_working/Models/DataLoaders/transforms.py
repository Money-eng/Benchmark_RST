import albumentations as A
from albumentations.pytorch import ToTensorV2

def get__val_test_img_transform():
    return A.Compose([
        ToTensorV2(),
    ])

def get_train_img_transform_1(patch_size=512):
    return A.Compose([
        A.RandomCrop(patch_size, patch_size), 
        ToTensorV2(),
    ])
    
def get_train_img_transform_2(patch_size=512):
    return A.Compose([
        A.RandomCrop(patch_size, patch_size), 
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    
def get_train_img_transform_3(patch_size=512):
    return A.Compose([
        A.RandomCrop(patch_size, patch_size),   
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])