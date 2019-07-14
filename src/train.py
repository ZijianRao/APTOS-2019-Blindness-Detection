import pandas as pd

import time
import torchvision
import torch.nn as nn
from tqdm import tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True

import data_loader

PRETRAINED_MODEL_PATH = '../../pytorch-pretrained-models'

def resnet50(pre_trained_path):
    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(pre_trained_path))
    num_features = model.fc.in_features
    # redefine last full connected layer
    model.fc = nn.Linear(num_features, 1)

    model = model.to(device)
    # only train those layers
    plist = [
            {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
            {'params': model.fc.parameters(), 'lr': 1e-3}
            ]
    return model, plist

def optimize(plist):
    optimizer = optim.Adam(plist, lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
    return optimizer, scheduler

def train(train_loader, valid_loader, model, optimizer, scheduler, criterion, num_epochs=15):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # update learning rate
        scheduler.step()
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
            tk0.set_postfix(loss=(running_loss / (counter * train_loader.batch_size)))
        epoch_loss = running_loss / len(train_loader)
        print('Training Loss: {:.4f}'.format(epoch_loss))

        valid_model(model, valid_loader, criterion)
    return model


def valid_model(model, valid_loader, criterion):
    avg_val_loss = 0.
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            avg_val_loss += criterion(outputs, labels).item() / len(valid_loader)
        print('Valid Loss: {:.4f}'.format(avg_val_loss))
        
    return avg_val_loss



def save_train(model):
    torch.save(model.state_dict(), "../data/model.bin")


def runner(pretrained_path, local=True):
    train_loader, valid_loader = data_loader.prepare_train(local)
    model, plist = resnet50(pretrained_path)
    optimizer, scheduler = optimize(plist)
    criterion = nn.MSELoss()
    model = train(train_loader, valid_loader, model, optimizer, scheduler, criterion)
    save_train(model)


def main():
    runner('../../pytorch-pretrained-models/resnet50-19c8e357.pth', local=True)

if __name__ == '__main__':
    main()