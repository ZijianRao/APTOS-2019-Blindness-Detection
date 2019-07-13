
import albumentations # data augmentation
from albumentations import torch as AT  
SIZE = 224

# some augmentations setup
# https://www.kaggle.com/artgor/basic-eda-and-baseline-pytorch-model
data_transforms = albumentations.Compose([
    albumentations.Resize(SIZE, SIZE),
    albumentations.HorizontalFlip(),
    albumentations.RandomBrightness(),
    albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(),
    AT.ToTensor()
    ])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(SIZE, SIZE),
    albumentations.Normalize(),
    AT.ToTensor()
    ])