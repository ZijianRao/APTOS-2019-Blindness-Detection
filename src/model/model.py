import pandas as pd
import numpy as np
import datetime
import collections
import time
import os
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision.models.resnet import  model_urls
from efficientnet_pytorch import EfficientNet


import data_loader
import config

device = torch.device("cuda:0")

class ModelHelper:
    """ Only for training"""
    def __init__(self, path=None, name=None):
        self.path = path # use path to load trained model
        self.name = name
        self.criterion = nn.MSELoss()
        self.model, self.plist = self.setup_model()
        self.optimizer, self.scheduler = self.prepare_optimizer_scheduler()
        self.best_score = 0.75
        self.data_dump = {}
        self.most_recent_loss = 0
    
    def setup_model(self):
        if 'efficientnet' in self.name:
            model = EfficientNet.from_pretrained(self.name)
        else:
            model_template = getattr(models, self.name)
            model = model_template(pretrained=False)

        if self.path is None:
            # load pretrained standard model
            if 'efficientnet' not in self.name:
                model_dict = torch.hub.load_state_dict_from_url(model_urls[self.name])
                model.load_state_dict(model_dict)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, 1)
            else:
                num_features = model._fc.in_features
                model._fc = nn.Linear(num_features, 1)
        else:
            # load trained by self model
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
            model.load_state_dict(torch.load(self.path))

        model = model.to(device)
        if self.path is not None:
        # only train those layers for trained model
            plist = [
                    {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
                    {'params': model.fc.parameters(), 'lr': 1e-3}
                    ]
        else:
            plist = model.parameters()
        return model, plist
    
    def prepare_optimizer_scheduler(self):
        optimizer = optim.Adam(self.plist, lr=0.001)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        return optimizer, scheduler

    def train_bucket(self, train_loader_iter, valid_loader_iter):
        valid_loader = list(valid_loader_iter)[0]
        for i, train_loader in enumerate(train_loader_iter):
            print(f'bucket {i}')
            self.train(train_loader, valid_loader, num_epochs=10, prefix=f'bucket_{i}')
            self.check_out_if_good(valid_loader, force_save=f'_bucket{i}')
        self.save_log()
    
    def train(self, train_loader, valid_loader, num_epochs=10, prefix=''):
        scheduler = self.scheduler
        optimizer = self.optimizer
        model = self.model
        criterion = self.criterion
        valid_score = 0.0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}', end=' ')
            # update learning rate
            scheduler.step(valid_score)
            # set into train mode, not eval mode. May impact: batchnorm and dropout
            model.train()
            running_loss = 0.0
            tk0 = tqdm(train_loader, total=int(len(train_loader)))
            counter = 0
            optimizer.zero_grad()   # clear accumulated gradient
            for bi, (inputs, labels) in enumerate(tk0):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    if (bi + 1) % 10 == 0:
                        optimizer.step()
                        optimizer.zero_grad()   # clear accumulated gradient
                running_loss += loss.item() * inputs.size(0)
                counter += 1
                tk0.set_postfix(epoch=epoch, loss=(running_loss / (counter * train_loader.batch_size)))

            # summary part
            average_loss = running_loss / (counter * train_loader.batch_size)
            self.most_recent_loss = average_loss
            epoch_loss = running_loss / len(train_loader)
            print('Training Loss: {:.4f}'.format(epoch_loss))
            valid_score, kappa_score = self.check_out_if_good(valid_loader)
            self.data_dump[f'{prefix}_epoch_{epoch}'] = (average_loss, valid_score, kappa_score)

        valid_score, kappa_score = self.check_out_if_good(valid_loader, force_save='final')
        return model
    
    def check_out_if_good(self, valid_loader, force_save=''):
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
        if force_save != '' or val_kappa > self.best_score * 0.95:
            self.best_score = max(val_kappa, self.best_score)
            name = f'{val_kappa:.2f}_{loss:.3f}_{self.most_recent_loss:.3f}_{self.name}' + force_save
            torch.save(model.state_dict(), os.path.join(config.CHECKOUT_PATH, name))
            print(f'Save checkpoint!')
        return loss, val_kappa
    
    def save_log(self):
        df = pd.DataFrame(self.data_dump).T
        name = f'{datetime.datetime.now()}_{self.name}.csv'
        df.to_csv(os.path.join(config.LOG_PATH, name))
        print(f'Log {name} saved!')


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