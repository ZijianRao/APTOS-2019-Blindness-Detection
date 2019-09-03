import pandas as pd
import os.path
import numpy as np
from joblib import Parallel, delayed
import psutil
import functools
from tqdm import tqdm
from PIL import Image
import joblib

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, StratifiedKFold

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import pickle


import data_transform
import config

def workflow_new():
    data = load_train_description()
    train_df = data[data['set'] != 'new'].reset_index(drop=True)
    valid_df = data[data['set'] == 'new'].reset_index(drop=True)

    train_data = load_from_cache('train_old', train_df)
    valid_data = load_from_cache('valid_old', valid_df)

    train_bucket = prepare_loader(train_data, train_df, data_transform.train_transform, num_workers=config.NUM_WORKERS)
    valid_bucket = prepare_loader(valid_data, valid_df, data_transform.valid_transform, num_workers=0)

    return train_bucket, valid_bucket

def workflow_mix():
    data = load_train_description()
    data = data[data['set'] != 'new']
    train_df, valid_df = generate_split(data, 0.2)
    # train_df, valid_df = generate_split(data, 0.25)

    train_data = load_from_cache('train_old', train_df)
    valid_data = load_from_cache('valid_old', valid_df)

    train_bucket = prepare_loader(train_data, train_df, data_transform.train_transform, num_workers=config.NUM_WORKERS)
    valid_bucket = prepare_loader(valid_data, valid_df, data_transform.valid_transform, num_workers=0)

    return train_bucket, valid_bucket


def generate_split(df, cutoff=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cutoff, random_state=42)
    X = df.index.values
    y = df['diagnosis'].values
    train_index, valid_index = next(sss.split(X, y))
    train_df = df.iloc[train_index].reset_index(drop=True)
    valid_df = df.iloc[valid_index].reset_index(drop=True)
    return train_df, valid_df


def cv_train_loader(cacheReset=False, fold=5):
    data = load_train_description()
    # use old as train
    # use new as valid
    df = data[data['set'] == 'new']

    kf = KFold(n_splits=fold)
    # kf = StratifiedKFold(n_splits=fold, random_state=42)
    
    data = load_from_cache('train_new', df)

    def list_indexer(data, index_array):
        return [data[i] for i in index_array]

    for train_index, valid_index in kf.split(df):
    # for train_index, valid_index in kf.split(df, y=df['diagnosis']):
        train = df.iloc[train_index]
        train_image = list_indexer(data, train_index)
        valid = df.iloc[valid_index]
        valid_image = list_indexer(data, valid_index)

        loader_tmp1 = prepare_loader(train_image, train, data_transform.train_transform, num_workers=config.NUM_WORKERS)
        loader_tmp2 = prepare_loader(valid_image, valid, data_transform.valid_transform, num_workers=0)

        yield loader_tmp1, loader_tmp2


def load_train_description():
    """ Load train label data for 15 and 19
    Meta data and data directly
    """
    old_train = pd.read_csv(config.PREV_DATA_PATH)
    old_test = pd.read_csv(config.PREV_TEST_PATH)
    train = pd.read_csv(config.CLEAN_TRAIN_DATA_PATH)
    df1 = train[['id_code', 'diagnosis']]
    df1['set'] = 'new'
    old_train['set'] = 'old_train'
    old_test = old_test[['image', 'level']]
    old_test['set'] = 'old_test'
    old_data = pd.concat([old_train, old_test])
    df2 = old_data.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
    df = pd.concat([df1, df2])
    df = df.reset_index(drop=True)
    df['path'] = df.apply(path_maker, axis=1)

    return df


def path_maker(row):
    id_code = row['id_code']
    datatype = row['set']
    if datatype == 'new':
        template = config.TRAIN_IMAGE_PATH
    elif datatype == 'old_train':
        template = config.PREV_TRAIN_IMAGE_PATH
    elif datatype == 'old_test':
        template = config.PREV_TEST_IMAGE_PATH
    return template.format(id_code)


def load_train_image(path_group, img_size=config.IMG_SIZE, adjustment=True):
    """ Load train images based on input path info """
    helper = functools.partial(image_reader, adjustment=adjustment, img_size=img_size)
    x_train = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(
        delayed(helper)(fp) for fp in path_group)
    return x_train


def image_reader(path, adjustment=False, img_size=config.IMG_SIZE):
    """ Read image resize it and return as numpy array"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = data_transform.crop_image_from_gray(image)
    # if adjustment:
        # only apply circle crop
    image = data_transform.circle_crop(image)
        
    image = cv2.resize(image, (img_size, img_size))
    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)

    image = image.astype(np.uint8)
    # image = transforms.ToPILImage()(image)

    return image


def prepare_loader(data, labels, transform, num_workers):
    dataset = RetinopathyDataset(labels, data, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers, shuffle=True, pin_memory=False)
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
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


def load_from_cache(name, df, reset=False, postfix='circleOnly'):
    name += str(config.IMG_SIZE) + postfix

    path = config.DATA_CACHE_PATH.format(name)
    if not reset and os.path.exists(path):
        data = joblib.load(path, mmap_mode='r')
        print(f'Load {name} from cache')
        print(data.shape)
    else:
        print(f'Redo {name}')
        data = load_train_image(df['path'].values, adjustment=False)
        data = np.stack(data)
        joblib.dump(data, path)
        for i in range(10):
            im = Image.fromarray(data[i])
            im.save(f'../data/processed_img/{name}_{i}.jpeg')
    return data

if __name__ == '__main__':
    # train_dataset = RetinopathyDatasetTrain(csv_file='../data/train.csv')
    # df = train_dataset.data
    # print(prepare_labels(df['diagnosis']))
    # image_reader('../data/previous/resized_train_15/10_left.jpg')
    # list(cv_train_loader(cacheReset=True))
    # print('done')
    # list(cv_train_loader())
    # workflow_mix()
    workflow_new()
