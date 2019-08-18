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
#     train_old()
    cv_train()

def train_old():
    model = EfficientNetModel(name='efficientnet-b3', fine_tune=False, lr=1e-4,
        path=os.path.join(config.CHECKOUT_PATH, '0.81_0.280_0.272_efficientnet-b3_old_all_no_circle'))
#     model = ModelHelper(name='resnet50', fine_tune=False)
    t, v = data_loader.workflow_mix()
    model.train(t, v, accum_gradient=2, n_freeze=0, num_epochs=7, name='old_all_no_circle', 
    )


def cv_train():
    for i, (train_loader, test_loader) in enumerate(data_loader.cv_train_loader(cacheReset=False)):
        obj = EfficientNetModel(name='efficientnet-b3', 
                          path=os.path.join(config.CHECKOUT_PATH, '0.81_0.280_0.272_efficientnet-b3_old_all_no_circle'),
                        #   path=os.path.join(config.CHECKOUT_PATH, '0.79_0.303_0.283_efficientnet-b3_old_all'),
                          fine_tune=False, lr=1e-3, opt_kappa=True)
        # obj = ModelHelper(name='resnet50', 
        #                   path=os.path.join(config.CHECKOUT_PATH, '0.76_0.350_0.308_resnet50_old_all'),
        #                   fine_tune=False)
                          
        obj.best_score = 0.85
        obj.train(train_loader, test_loader, accum_gradient=2, n_freeze=0, num_epochs=20, name=f'cv_{i}')
        pass

if __name__ == '__main__':
    main()