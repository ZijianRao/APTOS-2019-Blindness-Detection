import pandas as pd
import numpy as np
import scipy as sp
import datetime
import collections
import time
import os
from tqdm import tqdm
from functools import partial
from sklearn.metrics import cohen_kappa_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision.models.resnet import model_urls as resnet_model_urls
from torchvision.models.densenet import model_urls as densenet_model_urls
from efficientnet_pytorch import EfficientNet
from apex import amp


import data_loader
import config

device = torch.device("cuda:0")

class ModelHelper:
    """ Only for training"""
    def __init__(self, path=None, name=None, lr=1e-3, opt_kappa=False):
        self.path = path # use path to load trained model
        self.name = name
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.model, self.plist = self.setup_model()
        self.optimizer, self.scheduler = self.prepare_optimizer_scheduler()
        self.best_score = 0.75
        self.data_dump = {}
        self.most_recent_loss = 0
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        self.opt_kappa = opt_kappa
    
    
    def setup_model(self):
        model_template = getattr(models, self.name)
        model = model_template(pretrained=True)
        if 'res' in self.name:
            last_layer_name = 'fc'
        elif 'densenet' in self.name:
            last_layer_name = 'classifier'

        for param in model.parameters():
            param.requires_grad = False

        if self.path is None:
            # model_dict = torch.hub.load_state_dict_from_url(resnet_model_urls[self.name])
            # model.load_state_dict(model_dict)


            num_features = getattr(model, last_layer_name).in_features
            setattr(model, last_layer_name, nn.Linear(num_features, 1))
        else:
            num_features = getattr(model, last_layer_name).in_features
            setattr(model, last_layer_name, nn.Linear(num_features, 1))
            model.load_state_dict(torch.load(self.path))
            # no need to freeze
            # for param in model.parameters():
            #     param.requires_grad = False


        model = model.to(device)
        plist = model.parameters()

        return model, plist
    
    def prepare_optimizer_scheduler(self):
        optimizer = optim.Adam(self.plist, lr=self.lr)
        # optimizer = optim.SGD(self.plist, lr=self.lr, momentum=0.9)
        # optimizer = optim.SGD(self.plist, lr=self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
        return optimizer, scheduler


    def train(self, train_loader, valid_loader, accum_gradient=2, n_freeze=3, num_epochs=10, name=''):
        scheduler = self.scheduler
        optimizer = self.optimizer
        model = self.model
        criterion = self.criterion
        dump_kappa = [0]
        dump_valid = [1]
        valid_count = 0

        for epoch in range(1, num_epochs + 1):
            print(f'Epoch {epoch}/{num_epochs}')
            # Freeze the initial training to focus purely on linear part
            if epoch == (n_freeze+1):
                print('------------')
                print('Begin to train all parameters')
                print('------------')
                for param in model.parameters():
                    param.requires_grad = True
            # set into train mode, not eval mode. May impact: batchnorm and dropout
            model.train()
            running_loss = 0.0
            tk0 = tqdm(train_loader, total=int(len(train_loader)))
            counter = 0
            optimizer.zero_grad()   # clear accumulated gradient
            for bi, (inputs, labels) in enumerate(tk0):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (bi + 1) % accum_gradient == 0:
                    optimizer.step()
                    optimizer.zero_grad()   # clear accumulated gradient
                running_loss += loss.item() * inputs.size(0)
                counter += 1
                tk0.set_postfix(epoch=epoch, loss=(running_loss / (counter * train_loader.batch_size)))

            # handle tiny bug
            if (bi + 1) % accum_gradient != 0:
                optimizer.step()
                optimizer.zero_grad()   # clear accumulated gradient

            # summary part
            average_loss = running_loss / (counter * train_loader.batch_size)
            self.most_recent_loss = average_loss
            epoch_loss = running_loss / len(train_loader)
            print('Training Loss: {:.4f}'.format(epoch_loss))
            valid_score, kappa_score, kappa_opt, coefficients = self.valid_model(valid_loader)

            if kappa_score < dump_kappa[-1]:
                valid_count += 1
                print(f'Early stop {valid_count}/5')
            else:
                valid_count = 0

            if kappa_score > max(dump_kappa) or valid_score < min(dump_valid):
                self.check_out_valid(valid_score, kappa_score, name)

            if valid_count > 5:
                print('Early Stop!')
                # don't want to diverage too much
                break

            dump_kappa.append(kappa_score)
            dump_valid.append(valid_score)

            self.data_dump[f'{name}_epoch_{epoch}'] = (average_loss, valid_score, kappa_score, kappa_opt, coefficients)
            # update learning rate
            # scheduler.step(kappa_score)
            scheduler.step()

        self.save_log(name)
        return model
    
    def valid_model(self, valid_loader):
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
        y_pred_reg = np.concatenate(y_pred)
        y_pred = score_to_level(y_pred_reg.copy())
        y_valid = np.concatenate(y_valid)
        analyze_valid(y_valid, y_pred)
        val_kappa = cohen_kappa_score(y_valid, y_pred, weights='quadratic')

        if self.opt_kappa:
            optR = OptimizedRounder()
            optR.fit(y_pred_reg, y_valid)
            coefficients = optR.coefficients()
            y_pred_opt = optR.predict(y_pred_reg, coefficients)
            val_kappa_opt = cohen_kappa_score(y_valid, y_pred_opt, weights='quadratic')
        else:
            print('No optimized Kappa')
            val_kappa_opt = val_kappa
            coefficients = [0.5, 1.5, 2.5, 3.5]

        print(coefficients)
        print(f'Valid Kappa Loss: {val_kappa:.4f}, Optimized Kappa Loss: {val_kappa_opt:.4f}, MSE Loss: {loss:.4f}')
        return loss, val_kappa, val_kappa_opt, coefficients
    
    def save_log(self, name):
        df = pd.DataFrame(self.data_dump).T
        df.columns = ['Train_Loss', 'Valid_Loss', 'Valid_Kappa_Score', 'Optimized_Valid_Kappa_Score', 'Coefficients']
        name = f'{datetime.datetime.now()}_{self.name}_{name}.csv'
        df.to_csv(os.path.join(config.LOG_PATH, name))
        print(f'Log {name} saved!')

    def check_out_valid(self, valid_score, valid_kappa, postfix=''):
        name = f'{valid_kappa:.2f}_{valid_score:.3f}_{self.most_recent_loss:.3f}_{self.name}_{postfix}'
        torch.save(self.model.state_dict(), os.path.join(config.CHECKOUT_PATH, name))
        print(f'Save checkpoint!')

def analyze_valid(y_valid, y_pred):
    """ Provide auc for """
    print(classification_report(y_valid, y_pred, labels=[0, 1, 2, 3, 4]), flush=True)

def score_to_level(output, coef=config.CUTOFF_COEF):
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


#https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']