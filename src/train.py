import data_loader
from model.resnet import Resnet


def main():
    obj = Resnet(name='resnet101')
    obj.train()

if __name__ == '__main__':
    main()