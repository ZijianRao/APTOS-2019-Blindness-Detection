import numpy as np
import cv2
import albumentations # data augmentation
from albumentations import torch as AT  
SIZE = 224

import config

# some augmentations setup
# https://www.kaggle.com/artgor/basic-eda-and-baseline-pytorch-model
train_transform = albumentations.Compose([
    albumentations.HorizontalFlip(),
    albumentations.RandomBrightness(),
    albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(config.NORMALIZE),
    AT.ToTensor(),
    ])

test_transform = albumentations.Compose([
    albumentations.HorizontalFlip(),
    albumentations.Normalize(config.NORMALIZE),
    AT.ToTensor(),
    ])

def crop_image_from_gray(img,tol=7):
    """ Crop function copied from https://www.kaggle.com/chanhu/eye-inference-num-class-1-ver3"""
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img