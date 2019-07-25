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
#     t, v = data_loader.workflow_mix()

#     obj = ModelHelper(name='efficientnet-b4')
#     obj.train_bucket(t, v)
    cv_train()

def cv_train():
    for train_loader, test_loader in data_loader.cv_train_loader():
        obj = ModelHelper(name='efficientnet-b4', path=os.path.join(config.CHECKOUT_PATH, '0.79_0.623_0.495_efficientnet-b4'))
        obj.best_score = 0.85
        obj.train(train_loader, test_loader)

if __name__ == '__main__':
    main()