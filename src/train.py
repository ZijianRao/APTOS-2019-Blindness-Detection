import data_loader
import config
from model.model import ModelHelper 
from model.efficient_train import EfficientNetModel
import random
import os
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    seed_everything(42)
    train_old()
#     cv_train()

def train_old():
    t, v = data_loader.workflow_mix()
    model = EfficientNetModel(name='efficientnet-b4', fine_tune=False)
    model.train(t, v, accum_gradient=3, n_freeze=5, num_epochs=15, name='old_all')


def cv_train():
    for i, (train_loader, test_loader) in enumerate(data_loader.cv_train_loader()):
        obj = ModelHelper(name='efficientnet-b4', 
                          path=os.path.join(config.CHECKOUT_PATH, '0.79_0.318_0.212_efficientnet-b4_bucket_0'),
                          fine_tune=True)
                          
        obj.best_score = 0.85
        obj.train(train_loader, test_loader, name=f'cv_{i}')
        pass

if __name__ == '__main__':
    main()