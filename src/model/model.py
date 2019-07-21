import pandas as pd
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from sklearn.metrics import cohen_kappa_score

import data_loader
import config

device = torch.device("cuda:0")

class ModelRoot:
    """ Only for training"""
    def __init__(self, path, name=None):
        self.path = path
        self.name = name
        self.criterion = nn.MSELoss()
        self.model, self.plist = self.setup_model()
        self.optimizer, self.scheduler = self.prepare_optimizer_scheduler()
        self.best_score = 0.75
    
    def setup_model(self):
        raise NotImplementedError()
    
    def prepare_optimizer_scheduler(self):
        optimizer = optim.Adam(self.plist, lr=0.001)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        return optimizer, scheduler
    
    def train_bucket(self, train_loader_iter, valid_loader_iter):
        valid_loader = list(valid_loader_iter)[0]
        for i, train_loader in enumerate(train_loader_iter):
            print(f'bucket {i}')
            self.train(train_loader, valid_loader, num_epochs=10)
    
    def train(self, train_loader, valid_loader, num_epochs=10):
        scheduler = self.scheduler
        optimizer = self.optimizer
        model = self.model
        criterion = self.criterion
        valid_score = -1
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}', end=' ')
            # update learning rate
            scheduler.step(valid_score)
            # set into train mode, not eval mode. May impact: batchnorm and dropout
            model.train()
            running_loss = 0.0
            tk0 = tqdm(train_loader, total=int(len(train_loader)))
            counter = 0
            for bi, (inputs, labels) in enumerate(tk0):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                optimizer.zero_grad()   # clear accumulated gradient
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                counter += 1
                tk0.set_postfix(epoch=epoch, loss=(running_loss / (counter * train_loader.batch_size)))
            epoch_loss = running_loss / len(train_loader)
            print('Training Loss: {:.4f}'.format(epoch_loss))
            valid_score = self.check_out_if_good(epoch, valid_loader)

        return model
    
    def check_out_if_good(self, epoch, valid_loader):
        model = self.model
        criterion = self.criterion
        model.eval()
        y_pred = []
        y_valid = []
        running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                y_valid.append(labels)
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs_cpu = outputs.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
                y_pred.append(outputs_cpu)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                counter += 1
        loss = running_loss / (counter * valid_loader.batch_size)
        y_pred = np.concatenate(y_pred)
        y_pred = score_to_level(y_pred)
        y_valid = np.concatenate(y_valid)
        val_kappa = cohen_kappa_score(y_valid, y_pred, weights='quadratic')
        print(f'Valid Kappa Loss: {val_kappa:.4f}, MSE Loss: {loss:.4f}')
        if val_kappa > self.best_score:
            self.best_score = val_kappa
            torch.save(model.state_dict(), os.path.join(config.CHECKOUT_PATH, f'{val_kappa:.3f}_{self.name}_epoch_{epoch}'))
            print(f'Save checkpoint with loss: {val_kappa:.4f}')
        return val_kappa


def score_to_level(output):
    coef = config.CUTOFF_COEF
    for i, pred in enumerate(output):
        if pred < coef[0]:
            output[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            output[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            output[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            output[i] = 3
        else:
            output[i] = 4
    return output.astype(int)