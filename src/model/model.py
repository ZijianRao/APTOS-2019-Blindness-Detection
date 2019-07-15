import pandas as pd

import time
import torchvision
import torch.nn as nn
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os
from sklearn.metrics import cohen_kappa_score

import data_loader

device = torch.device("cuda:0")

class ModelRoot:
    """ Only for training"""
    def __init__(self, path, local=True):
        self.path = path
        self.name = None
        self.train_loader, self.valid_loader = data_loader.prepare_train(local=local)
        self.criterion = nn.MSELoss()
        self.model, self.plist = self.setup_model()
        self.optimizer, self.scheduler = self.prepare_optimizer_scheduler()
    
    def setup_model(self):
        raise NotImplementedError()
    
    def prepare_optimizer_scheduler(self):
        optimizer = optim.Adam(self.plist, lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
        return optimizer, scheduler
    
    def train(self):
        pass

    
