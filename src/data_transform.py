import numpy as np
import cv2
from torchvision import transforms
import config

train_transform = transforms.Compose([
    # transforms.Resize(config.IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation((-120, 120)),
    transforms.CenterCrop(config.IMG_SIZE),
    # transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.5, 1.0)),
    transforms.ToTensor(),
    # transforms.Normalize(*config.NORMALIZE)
    ])

valid_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation((-120, 120)),
    # transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(*config.NORMALIZE)
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
    
def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    
    img = crop_image_from_gray(img)    

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 