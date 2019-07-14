import pandas as pd
import numpy as np

import torchvision
import torch.nn as nn
from tqdm import tqdm

import torch
from torchvision import transforms
import os

import data_loader
import config

device = torch.device("cuda:0")

def resnet50(model_path):
    model = torchvision.models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    # redefine last full connected layer
    model.fc = nn.Linear(num_features, 1)
    pre_trained_path = os.path.join(model_path)
    model.load_state_dict(torch.load(pre_trained_path))

    model = model.to(device)
    return model

def run_test(model, test_loader, num_samples, TTA=3):
    test_pred = np.zeros((num_samples, 1))
    model.eval()

    for _ in range(TTA):
        with torch.no_grad():
            for i, (inputs, _) in tqdm(enumerate(test_loader), total=int(len(test_loader))):
                inputs = inputs.to(device, dtype=torch.float)
                pred = model(inputs)
                pred = pred.detach().cpu().squeeze().numpy().reshape(-1, 1)
                test_pred[i * config.BATCH_SIZE : i * config.BATCH_SIZE + len(pred)] += pred
            
    output = test_pred / TTA

    return output

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
    return output

def prepare_submission(output, id_code):
    submission = pd.DataFrame({'id_code': id_code, 'diagnosis':np.squeeze(output).astype(int)})
    return submission

def workflow(local):
    model_path = '../data/model.bin'
    test_loader, test_data = data_loader.prepare_test(local=local)
    model = resnet50(model_path)
    output = run_test(model, test_loader, len(test_data), TTA=2)
    output = score_to_level(output)
    submission = prepare_submission(output, test_data['id_code'].values)
    submission.to_csv('../data/submission.csv', index=False)

if __name__ == '__main__':
    workflow(True)