import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetWrapper(nn.Module):
    """
    EfficientNet-B3 pretrained on ImageNet, adapted for medical image classification.
    Grayscale images are converted to 3 channels to match EfficientNet's input.

    Usage:
        model = EfficientNetWrapper(num_classes=3, unfreeze_layers=2)
    """
    def __init__(self, num_classes=3, unfreeze_layers=2, dropout=0.3):
        super(EfficientNetWrapper, self).__init__()

        # loading pretrained EfficientNet-B3
        self.model= models.efficientnet_b3(weights="IMAGENET1K_V1")

        for param in self.model.parameters():
            param.requires_grad= False

        layers= list(self.model.features.children())
        for layer in layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad= True

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward (self,x):
        if x.shape[1]==1:
            x= x.repeat(1, 3, 1, 1)     #(batch, channels, height, width)

        x= self.model(x)
        return torch.sigmoid(x)
    

class DenseNetWrapper(nn.Module):
    """
    DenseNet-121 pretrained on ImageNet, adapted for medical image classification.
    Particularly effective for chest X-ray tasks.

    Usage:
        model = DenseNetWrapper(num_classes=3, unfreeze_layers=2)
    """
    def __init__(self, num_classes=3, unfreeze_layers=2, dropout=0.3):
        super(DenseNetWrapper, self).__init__()
        self.model = models.densenet121 (weights="IMAGENET1K_V1")
        
        for param in self.model.parameters():
            param.requires_grad= False

        layers = list(self.model.features.children())
        for layer in layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.model(x)
        return torch.sigmoid(x)


class ResNetWrapper(nn.Module):
    """
    ResNet-50 pretrained on ImageNet, adapted for medical image classification.
    Good general purpose backbone for both X-ray and MRI tasks.

    Usage:
    model = ResNetWrapper(num_classes=3, unfreeze_layers=2)
    """
    def __init__(self, num_classes=3, unfreeze_layers=2, dropout=0.3):
        super(ResNetWrapper, self).__init__()

        self.model= models.resnet50(weights="IMAGENET1K_V1")
        
        for param in self.model.parameters():
            param.requires_grad= False

        layers = list(self.model.children())
        for layer in layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.model(x)
        return torch.sigmoid(x)