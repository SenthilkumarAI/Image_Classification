from torchvision import models
from torch import nn
n_class = 2


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad=False
    return model

def check_requires_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)



def classification_model(n_class):
    mobilenet_model = models.mobilenet_v2(pretrained=False)

    mobilenet_model.classifier[1] = nn.Linear(in_features=1280, out_features=n_class, bias=True)

    resnet50_model = models.resnet50(pretrained=False)
    resnet50_model.fc = nn.Linear(2040, n_class, bias=True)


    vgg16_model = models.vgg16(pretrained=True)
    vgg16_model = freeze_model(vgg16_model)

    vgg16_model.classifier[0] = nn.Linear(25088, 4096, bias=True)
    vgg16_model.classifier[1] = nn.ReLU(inplace=True)
    vgg16_model.classifier[2] = nn.Dropout(p=0.5, inplace=False)
    vgg16_model.classifier[3] = nn.Linear(4096, 4096, bias=True)
    vgg16_model.classifier[4] = nn.ReLU(inplace=True)
    vgg16_model.classifier[5] = nn.Dropout(p=0.5, inplace=False)
    vgg16_model.classifier[6] = nn.Linear(4096, n_class, bias=True)

    check_requires_grad(vgg16_model)


    return vgg16_model, resnet50_model, mobilenet_model

if __name__ == "__main__":
    vgg_16, resnet50, mobilenet = classification_model(1)
    print(vgg_16)