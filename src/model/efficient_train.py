from efficientnet_pytorch import EfficientNet

from model.model import ModelHelper
import torch
import torch.nn as nn

device = torch.device("cuda:0")

class EfficientNetModel(ModelHelper):
    def __init__(self, path=None, name=None, fine_tune=True, lr=1e-3, opt_kappa=False):
        super().__init__(path=path, name=name, fine_tune=fine_tune, lr=lr, opt_kappa=opt_kappa)
    
    def setup_model(self):
        model = EfficientNet.from_pretrained(self.name)

        if self.path is None:
            # load pretrained standard model
            # freeze all parameters first
            num_features = model._fc.in_features
        else:
            num_features = model._fc.in_features
            model._fc = nn.Linear(num_features, 1)
            model.load_state_dict(torch.load(self.path))

        for param in model.parameters():
            param.requires_grad = False
        model._fc = nn.Linear(num_features, 1)

        model = model.to(device)
        if self.fine_tune:
            plist = [
                    {'params': model._fc.parameters(), 'lr': self.lr}
            ]
        else:
            plist = model.parameters()

        return model, plist
