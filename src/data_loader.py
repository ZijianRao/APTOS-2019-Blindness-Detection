import pandas as pd
import os.path
import numpy as np
from joblib import Parallel, delayed
import psutil
import functools
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import pickle


import data_transform
import config

def workflow_mix():
    data = load_train_description()
    # use old as train
    # use new as valid
    train_df = data[data['set'] == 'old']
    valid_df = data[data['set'] == 'new']

    train_bucket_iter = prepare_bucket(train_df, bucket_size=config.BUCKET_SPLIT, adjustment=False)
    valid_bucket_iter = prepare_bucket(valid_df, bucket_size=1, adjustment=True)
    return train_bucket_iter, valid_bucket_iter

def workflow_train():
    data = load_train_description()
    # use old as train
    # use new as valid
    train_df = data[data['set'] == 'new']
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train = train_df.iloc[:2900]
    valid = train_df.iloc[2900:]

    train_loader = list(prepare_bucket(train, bucket_size=1, adjustment=True))[0]
    valid_loader = list(prepare_bucket(valid, bucket_size=1, adjustment=True))[0]
    return train_loader,valid_loader 

def load_train_description():
    """ Load train label data for 15 and 19
    Meta data and data directly
    """
    old_train = pd.read_csv(config.PREV_DATA_PATH)
    train = pd.read_csv(config.CLEAN_TRAIN_DATA_PATH)
    df1 = train[['id_code', 'diagnosis']]
    df1['set'] = 'new'
    old_train['set'] = 'old'
    df2 = old_train.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
    df = pd.concat([df1, df2])
    df = df.reset_index(drop=True)
    df['path'] = df.apply(path_maker, axis=1)

    return df


def path_maker(row):
    id_code = row['id_code']
    datatype = row['set']
    if datatype == 'new':
        template = config.TRAIN_IMAGE_PATH
    else:
        template = config.PREV_TRAIN_IMAGE_PATH
    return template.format(id_code)


def load_train_image(path_group, img_size=config.IMG_SIZE, adjustment=True):
    """ Load train images based on input path info
    
    Args:
        path_group ([type]): [description]
        img_size ([type], optional): [description]. Defaults to config.IMG_SIZE.
        adjustment (bool, optional): [description]. Defaults to True.
    """
    helper = functools.partial(image_reader, adjustment=adjustment, img_size=img_size)
    x_train = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(
        delayed(helper)(fp) for fp in path_group)
    return x_train


def image_reader(path, adjustment=True, img_size=config.IMG_SIZE):
    """ Read image resize it and return as PIL image"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if adjustment:
        image = data_transform.crop_image_from_gray(image)
        image = cv2.resize(image, (img_size, img_size))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 30), -4, 128)
    else:
        image = cv2.resize(image, (img_size, img_size))

    image = transforms.ToPILImage()(image)
    return image


def prepare_bucket(train_df, bucket_size=config.BUCKET_SPLIT, adjustment=False):
    for df in np.array_split(train_df, bucket_size):
        data = load_train_image(df['path'].values, adjustment=False)
        yield prepare_loader(data, df, data_transform.test_transform)


def prepare_loader(data, labels, transform):
    dataset = RetinopathyDataset(labels, data, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
    return data_loader


class RetinopathyDataset(Dataset):
    """ For training purpose """

    def __init__(self, labels, image_group, transform=None):
        self.data = labels
        self.image_group = image_group
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.image_group[idx]
        label = self.data['diagnosis'].values[idx]
        label = np.expand_dims(label, -1)

        if self.transform is not None:
            image = self.transform(image)

        return image, label



###############DUMP
def prepare_sampler(df):
    tr, val = train_test_split(df, stratify=df, test_size=config.VALIDATION_SIZE)
    train_sampler = SubsetRandomSampler(list(tr.index))
    valid_sampler = SubsetRandomSampler(list(val.index))
    return train_sampler, valid_sampler

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
    # train_dataset = RetinopathyDatasetTrain(csv_file='../data/train.csv')
    # df = train_dataset.data
    # print(prepare_labels(df['diagnosis']))
    image_reader('../data/previous/resized_train_15/10_left.jpg')