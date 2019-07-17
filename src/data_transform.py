import numpy as np
import cv2
from torchvision import transforms
import config

train_transform = transforms.Compose([
    transforms.RandomAffine((-120, 120)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    # transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.5, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(*config.NORMALIZE)])

test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize(*config.NORMALIZE)
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