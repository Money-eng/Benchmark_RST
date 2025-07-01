import albumentations as A
from albumentations.pytorch import ToTensorV2

# original size of 2D grayscale images is 1348 × 1166 - need to add padding to make % 32 == 0 (height and width) - padding on right and bottom
height = 1348
width = 1166

ajusted_height = height + (32 - (height % 32)) if height % 32 != 0 else height
ajusted_width = width + (32 - (width % 32)) if width % 32 != 0 else width


def get__val_test_img_transform():
    global ajusted_height, ajusted_width
    return A.Compose([
        A.PadIfNeeded(min_height=ajusted_width, min_width=ajusted_height, border_mode=0, position='top_left'), # TODO : handle opposition
        ToTensorV2(),
    ])

def get_train_img_transform_1(patch_size=512):
    global ajusted_height, ajusted_width
    if patch_size is None:
        return A.Compose([
            A.PadIfNeeded(min_height=ajusted_width, min_width=ajusted_height, border_mode=0, position='top_left'),
            ToTensorV2(),
        ])
    else: 
        return A.Compose([
            A.RandomCrop(patch_size, patch_size), 
            ToTensorV2(),
        ])
    
def get_train_img_transform_2(patch_size=512):
    global ajusted_height, ajusted_width
    if patch_size is None:
        return A.Compose([
            A.PadIfNeeded(min_height=ajusted_width, min_width=ajusted_height, border_mode=0, position='top_left'),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.RandomCrop(patch_size, patch_size), 
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    
def get_train_img_transform_3(patch_size=512):
    global ajusted_height, ajusted_width
    if patch_size is None:
        return A.Compose([
            A.PadIfNeeded(min_height=ajusted_width, min_width=ajusted_height, border_mode=0, position='top_left'),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.RandomCrop(patch_size, patch_size),   
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    
def get_train_serie_transform():
    global ajusted_height, ajusted_width
    return A.Compose([
        A.PadIfNeeded(min_height=ajusted_width, min_width=ajusted_height, border_mode=0, position='top_left'),
        ToTensorV2(),
    ])