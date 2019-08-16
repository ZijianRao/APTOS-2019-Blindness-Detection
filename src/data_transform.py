import numpy as np
import cv2
# from torchvision import transforms
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, Flip
from albumentations.pytorch import ToTensor
from skimage.color import rgb2gray,rgba2rgb
import PIL.Image


import config


train_transform = Compose([
    Flip(always_apply=True),
    ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=(0, 0.35),
        rotate_limit=365,
        p=1.0),
    RandomBrightnessContrast(p=1.0),
    # progressive resize exp ?
    ToTensor()
])

valid_transform = Compose([
    ToTensor()
])


# train_transform = transforms.Compose([
#     # transforms.Resize(config.IMG_SIZE),
#     transforms.RandomHorizontalFlip(p=0.2),
#     transforms.RandomVerticalFlip(p=0.2),
#     transforms.RandomRotation((-120, 120)),
#     # transforms.CenterCrop(config.IMG_SIZE),
#     transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),  # just mimic center zoom crop
#     transforms.ToTensor(),
#     # transforms.Normalize(*config.NORMALIZE)
#     ])

# valid_transform = transforms.Compose([
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomRotation((-120, 120)),
#     # transforms.Resize(config.IMG_SIZE),
#     transforms.ToTensor(),
#     # transforms.Normalize(*config.NORMALIZE)
# ])

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
            img = np.stack([img1,img2,img3],axis=-1)
        return img

    
def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
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

def estimate_radius(img):
    mx = img[img.shape[0] // 2,:,:].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2
    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2
    return (ry, rx)

def subtract_gaussian_blur(img,b=5):
    gb_img = cv2.GaussianBlur(img, (0, 0), b)
    return cv2.addWeighted(img, 4, gb_img, -4, 128)

def remove_outer_circle(a, p, r):
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)
    return a * b + 128 * (1 - b)

def crop_img(img, h, w):
    h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
    w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0
    crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]
    return crop_img

def place_in_square(img, r, h, w):
    new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    new_img += 128
    new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img
    return new_img

def prepare_rgba(img, img_size=config.IMG_SIZE):
    r = img_size // 2
    ry, rx = estimate_radius(img)
    resize_scale = r / max(rx, ry)
    w = min(int(rx * resize_scale * 2), r*2)
    h = min(int(ry * resize_scale * 2), r*2)
    img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
    img = crop_img(img, h, w)
    #make rgba data for training
    img_rgba = np.zeros([img.shape[0],img.shape[1],4])
    for row in range(4):
        img2 = subtract_gaussian_blur(img,(row+1)*5)
        img_rgba[:,:,row] = rgb2gray(img2)
    # img = remove_outer_circle(img_rgba, 0.9, r)
    img = cv2.resize(img, (img_size, img_size))


    return img


def prepare_rgb(img, img_size=config.IMG_SIZE):
    img = prepare_rgba(img, img_size=img_size)
    cv2.imwrite('../data/processed_img/color_img.jpg', img)

    img = PIL.Image.fromarray(img)
    img = img.convert('RGB')

    return img

if __name__ == '__main__':
    img = cv2.imread('../data/train_images/ffd97f8cd5aa.png')
    img = prepare_rgb(img)
    img.save('../data/processed_img/test.png')
    print(img)