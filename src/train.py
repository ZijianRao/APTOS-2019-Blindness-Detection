import data_loader
import config
from model.resnet import Resnet
from model.resnext import Resnext
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
    # t, v = data_loader.workflow_train()

    obj = Resnet(name='resnet50')
    # obj = Resnext(path=os.path.join(config.CHECKOUT_PATH, '0.837_resnext50_32x4d_epoch_6'))
    obj.train_bucket(t, v)
    obj.train(t, v, num_epochs=10)

if __name__ == '__main__':
    main()