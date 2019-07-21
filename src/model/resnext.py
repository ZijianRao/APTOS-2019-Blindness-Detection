import torch
import torch.nn as nn
import pretrainedmodels

import torchvision.models as models
from torchvision.models.resnet import  model_urls

from model.model import ModelRoot
import config

device = torch.device("cuda:0")


class Resnext(ModelRoot):
    def __init__(self, name='resnext50_32x4d', path=None):
        super().__init__(path, name)

    def setup_model(self):
        model = models.resnext50_32x4d(pretrained=False)
        if self.path is None:
            model_dict = torch.hub.load_state_dict_from_url(model_urls[self.name])
            model.load_state_dict(model_dict)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
        else:
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)
            model.load_state_dict(torch.load(self.path))

        model = model.to(device)
        # only train those layers
        if self.path is not None:
            plist = [
                    {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
                    {'params': model.fc.parameters(), 'lr': 1e-3}
                    ]
        else:
            plist = model.parameters()
        return model, plist