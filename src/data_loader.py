import pandas as pd
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


from PIL import Image, ImageFile
import cv2


import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import data_transform
import config




class RetinopathyDataset(Dataset):

    def __init__(self, data, image_path_maker, transform=None, datatype='train'):
        self.data = data
        self.transform = transform
        self.datatype = datatype 
        self.image_path_maker = image_path_maker

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data['id_code'].values[idx]
        image_path = self.image_path_maker(image_name)
        image = image_reader(image_path)

        label = self.data['diagnosis'].values[idx]
        label = np.expand_dims(label, -1)

        if self.transform is not None:
            image = self.transform(image=image)
            image = image['image']

        return image, label


def image_reader(path):
    """ Read image resize it and return as PIL image"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = data_transform.crop_image_from_gray(image)
    image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 30), -4, 128)
    # image = transforms.ToPILImage()(image)
    return image


def image_path_maker(image_name):
    img_path = os.path.join(config.DATA_PATH, config.TRAIN_DATA_FOLDER, image_name + '.png')
    return img_path


def prepare_train():
    """ Prepare train, validation data """
    train_data = pd.read_csv(config.TRAIN_DATA_PATH)
    dataset = RetinopathyDataset(train_data, image_path_maker, transform=data_transform.test_transform, datatype='train')
    train_loader, valid_loader = prepare_data_loader(dataset, train_data['diagnosis'])
    return train_loader, valid_loader


def prepare_data_loader(dataset, df):
    train_sampler, valid_sampler = prepare_sampler(df)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=valid_sampler)
    return train_loader, valid_loader


def prepare_sampler(df):
    tr, val = train_test_split(df, stratify=df, test_size=config.VALIDATION_SIZE)
    train_sampler = SubsetRandomSampler(list(tr.index))
    valid_sampler = SubsetRandomSampler(list(val.index))
    return train_sampler, valid_sampler

###############DUMP
def prepare_labels(y):
    """ CLASSIFICATION SETUP: Transform 5 scores to vector representation"""
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder

if __name__ == '__main__':
    train_dataset = RetinopathyDatasetTrain(csv_file='../data/train.csv')
    df = train_dataset.data
    print(prepare_labels(df['diagnosis']))