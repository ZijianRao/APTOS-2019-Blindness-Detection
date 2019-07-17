import data_loader
from model.resnet import Resnet
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
    obj = Resnet(name='resnet50')
    obj.train(num_epochs=15)

if __name__ == '__main__':
    main()