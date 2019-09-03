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
#     model = EfficientNetModel(name='efficientnet-b3', lr=1e-3)
#     model = EfficientNetModel(name='efficientnet-b4', lr=1e-3)
#     model = EfficientNetModel(name='efficientnet-b0', lr=1e-3)
    model = EfficientNetModel(name='efficientnet-b5', lr=1e-3)
#     model = ModelHelper(name='densenet121', lr=1e-3)
#     model = ModelHelper(name='resnet50', lr=1e-3)

    t, v = data_loader.workflow_new()
    model.train(t, v, accum_gradient=5, n_freeze=0, num_epochs=20, name='old_all_circle_only')


def cv_train():
    for i, (train_loader, test_loader) in enumerate(data_loader.cv_train_loader(cacheReset=False)):
        # obj = EfficientNetModel(name='efficientnet-b3', 
        #                   path=os.path.join(config.CHECKOUT_PATH, '0.89_0.387_0.331_efficientnet-b3_old_all_circle_only'),
        #                   lr=1e-4, opt_kappa=False)
        # obj = EfficientNetModel(name='efficientnet-b4', 
        #                   path=os.path.join(config.CHECKOUT_PATH, '0.81_0.275_0.232_efficientnet-b4_old_all_no_circle'),
        #                   lr=1e-3, opt_kappa=False)
        obj = ModelHelper(name='densenet121', 
                          path=os.path.join(config.CHECKOUT_PATH, '0.87_0.392_0.425_densenet121_old_all_circle_only'),
                          lr=1e-4, opt_kappa=False)
        # obj = ModelHelper(name='resnet50', 
        #                   path=os.path.join(config.CHECKOUT_PATH, '0.79_0.330_0.332_resnet50_old_all_no_circle'),
        #                   lr=1e-2, opt_kappa=False)
                          
        obj.best_score = 0.85
        obj.train(train_loader, test_loader, accum_gradient=3, n_freeze=2, num_epochs=12, name=f'fine_tune_cv_{i}')
        pass

if __name__ == '__main__':
    main()