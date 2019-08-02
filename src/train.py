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
    path = os.path.join(config.CHECKOUT_PATH, '0.78_0.325_0.252_efficientnet-b4_bucket_0')
    ModelHelper.train_bucket('efficientnet-b4', t, v, path, fine_tune=False)
#     cv_train()

def cv_train():
    for i, (train_loader, test_loader) in enumerate(data_loader.cv_train_loader()):
        # obj = ModelHelper(name='efficientnet-b4', 
        #                   path=os.path.join(config.CHECKOUT_PATH, '0.74_0.387_0.394_efficientnet-b4finalbucket_14'),
        #                   fine_tune=True)
                          
        # obj.best_score = 0.85
        # obj.train(train_loader, test_loader, name=f'cv_{i}')
        pass

if __name__ == '__main__':
    main()