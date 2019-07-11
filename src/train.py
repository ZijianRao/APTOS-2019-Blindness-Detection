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

def main():
    train_loader, valid_loader = data_loader.prepare_data()

    model = torchvision.models.resnet101(pretrained=False)
    pre_trained_path = os.path.join(PRETRAINED_MODEL_PATH, 'resnet101-5d3b4d8f.pth')
    model.load_state_dict(torch.load(pre_trained_path))
    num_features = model.fc.in_features
    # one hot representation
    # model.fc = nn.Linear(num_features, 5)
    # regression problem
    model.fc = nn.Linear(num_features, 1)

    model = model.to(device)

    # only train those layers
    plist = [
            {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
            {'params': model.fc.parameters(), 'lr': 1e-3}
            ]

    optimizer = optim.Adam(plist, lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10)

    since = time.time()
    criterion = nn.MSELoss()
    num_epochs = 15
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        model.train()
        running_loss = 0.0
        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        counter = 0
        for bi, d in enumerate(tk0):
            inputs = d["image"]
            labels = d["labels"].view(-1, 1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model.state_dict(), "../data/model.bin")

if __name__ == '__main__':
    main()