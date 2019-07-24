import data_loader
import config
from model.model import ModelHelper 
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
    t, v = data_loader.workflow_mix()

    obj = ModelHelper(name='efficientnet-b4')
    # t, v = data_loader.workflow_train()
    # obj = Resnext(path=os.path.join(config.CHECKOUT_PATH, '0.837_resnext50_32x4d_epoch_6'))
    obj.train_bucket(t, v)
    # obj.train(t, v, num_epochs=10)
#     cv_train()

def cv_train():
    for train_loader, test_loader in data_loader.cv_train_loader():
        obj = ModelHelper(name='resnet50', path=os.path.join(config.CHECKOUT_PATH, '0.69_0.830_0.622_resnet50'))
        obj.best_score = 0.9
        obj.train(train_loader, test_loader)

if __name__ == '__main__':
    main()