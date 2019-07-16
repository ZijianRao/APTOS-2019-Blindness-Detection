import torchvision
import torch
import torch.nn as nn

from model.model import ModelRoot
import config

device = torch.device("cuda:0")

class Resnet(ModelRoot):
    def __init__(self, name='resnet50', local=True):
        path = config.PRETRAINED_PATH[name]
        super().__init__(path, name, local)

    def setup_model(self):
        model = getattr(torchvision.models, self.name)
        model = model(pretrained=False)
        model.load_state_dict(torch.load(self.path))
        num_features = model.fc.in_features
        # redefine last full connected layer
        model.fc = nn.Linear(num_features, 1)

        model = model.to(device)
        # only train those layers
        plist = [
                {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
                {'params': model.fc.parameters(), 'lr': 1e-3}
                ]
        plist = model.parameters()
        return model, plist