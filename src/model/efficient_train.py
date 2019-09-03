from efficientnet_pytorch import EfficientNet

from model.model import ModelHelper
import torch
import torch.nn as nn

device = torch.device("cuda:0")

class EfficientNetModel(ModelHelper):
    def __init__(self, path=None, name=None, lr=1e-3, opt_kappa=False):
        super().__init__(path=path, name=name, lr=lr, opt_kappa=opt_kappa)
    
    def setup_model(self):
        model = EfficientNet.from_pretrained(self.name)
        for param in model.parameters():
            param.requires_grad = False

        if self.path is None:
            # load pretrained standard model
            # freeze all parameters first
            num_features = model._fc.in_features
            model._fc = nn.Linear(num_features, 1)
        else:
            num_features = model._fc.in_features
            model._fc = nn.Linear(num_features, 1)
            model.load_state_dict(torch.load(self.path))


        model = model.to(device)
        plist = model.parameters()

        return model, plist
