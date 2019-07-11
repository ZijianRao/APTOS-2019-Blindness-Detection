import pandas as pd
import os.path
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


from PIL import Image, ImageFile
import cv2


import torch
from torch.utils.data import Dataset
from torchvision import transforms

import data_transform

DATA_PATH = '../data/'
TRAIN_DATA_FOLDER = 'train_images'

TRAIN_DATA_PATH = '../data/train.csv'

class RetinopathyDataset(Dataset):

    def __init__(self, transform, csv_file, data_type='train'):

        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(DATA_PATH, TRAIN_DATA_FOLDER, self.data.loc[idx, 'id_code'] + '.png')
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image=img)
        image = image['image']

        label = torch.tensor(self.data.loc[idx, 'diagnosis'])

        # TODO: missing test part

        return {'image': image,
                'labels': label}

def prepare_data():
    """ Prepare train, validation, and test data """
    dataset = RetinopathyDataset(transform=data_transform.data_transforms, csv=TRAIN_DATA_PATH, datatype='train')
    test_set = RetinopathyDataset(df=test, datatype='test', transform=data_transforms_test)
    tr, val = train_test_split(train.diagnosis, stratify=train.diagnosis, test_size=0.1)
    train_sampler = SubsetRandomSampler(list(tr.index))
    valid_sampler = SubsetRandomSampler(list(val.index))


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