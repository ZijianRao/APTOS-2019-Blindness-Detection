import pandas as pd
import os.path
import numpy as np
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

def generateTrainArray(img_size=config.IMG_SIZE):
    path = config.LOCAL_TRAIN_IMAGE_ARRAY_PATH.format(img_size)

    if os.path.exists(path):
        x_train = pickle.load(open(path, 'rb'))
    else:
        train_data = pd.read_csv(config.LOCAL_TRAIN_DATA_PATH)
        N = train_data.shape[0]
        maker = image_path_maker('train', local=True)
        # x_train = np.empty((N, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)

        id_codes = train_data['id_code'].values.tolist()
        x_train = []
        for i, id_code in tqdm(enumerate(id_codes), total=N):
            image_path = maker(id_code)
            image = image_reader(image_path)
            x_train.append(image)
            # image = np.array(image, dtype=np.uint8)
            # x_train[i, :, :, :] = image
            # image = Image.fromarray(image)
        pickle.dump(x_train, open(path, 'wb'))
    return x_train



class RetinopathyDataset(Dataset):

    def __init__(self, data, image_path_maker, transform=None, datatype='train', cache=None):
        self.data = data
        self.transform = transform
        self.datatype = datatype 
        self.image_path_maker = image_path_maker
        self.cache = cache

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache is None:
            image_name = self.data['id_code'].values[idx]
            image_path = self.image_path_maker(image_name)
            image = image_reader(image_path)
        else:
            image = self.cache[idx]

        if self.datatype == 'train':
            label = self.data['diagnosis'].values[idx]
            label = np.expand_dims(label, -1)
        else:
            label = 0 

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def image_reader(path, img_size=config.IMG_SIZE):
    """ Read image resize it and return as PIL image"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = data_transform.crop_image_from_gray(image)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 30), -4, 128)
    image = transforms.ToPILImage()(image)
    return image


def image_path_maker(datatype='train', local=True):
    if datatype == 'train':
        if local:
            path_prefix = config.LOCAL_TRAIN_IMAGE_PATH
        else:
            path_prefix = config.REMOTE_TRAIN_IMAGE_PATH
    else:
        if local:
            path_prefix = config.LOCAL_TEST_IMAGE_PATH
        else:
            path_prefix = config.REMOTE_TEST_IMAGE_PATH

    def image_path_maker_helper(image_name):
        img_path = os.path.join(path_prefix, image_name + '.png')
        return img_path
    return image_path_maker_helper 


def prepare_train(local=True):
    """ Prepare train, validation data """
    if local == True:
        train_data = pd.read_csv(config.LOCAL_TRAIN_DATA_PATH)
    else:
        train_data = pd.read_csv(config.REMOTE_TRAIN_DATA_PATH)
    
    cache = generateTrainArray()

    dataset = RetinopathyDataset(train_data, image_path_maker('train', local=local), transform=data_transform.test_transform, datatype='train', cache=cache)
    train_loader, valid_loader = prepare_data_loader(dataset, train_data['diagnosis'])
    return train_loader, valid_loader

def prepare_test(local=True):
    """ Prepare test data """
    if local == True:
        test_data = pd.read_csv(config.LOCAL_TEST_DATA_PATH)
    else:
        test_data = pd.read_csv(config.REMOTE_TEST_DATA_PATH)

    dataset = RetinopathyDataset(test_data, image_path_maker('test', local=local), transform=data_transform.test_transform, datatype='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False)
    return test_loader, test_data


def prepare_data_loader(dataset, df):
    train_sampler, valid_sampler = prepare_sampler(df)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=config.NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=valid_sampler, num_workers=config.NUM_WORKERS)
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
    # train_dataset = RetinopathyDatasetTrain(csv_file='../data/train.csv')
    # df = train_dataset.data
    # print(prepare_labels(df['diagnosis']))
    generateTrainArray(384)