import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv2d -> BatchNorm -> ReLU -> MaxPool
    This is the basic building unit of our CNN.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock,self).__init__()

        self.block= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self,x):
        return self.block(x)
    
class CustomCNN(nn.Module):
    """
    Configurable CNN for medical image classification.
    Supports both multi-class and multi-label outputs.

    Usage:
        model = CustomCNN(num_classes=3, num_blocks=4, base_filters=32, dropout=0.3)
    """
    def __init__(self, num_classes=3, num_blocks=4, base_filters=32, dropout=0.3):
        super(CustomCNN,self).__init__()

        layers=[]
        in_channels=1     #grayscale

        for i in range(num_blocks):
            out_channels= base_filters *(2**i)
            layers.append(ConvBlock(in_channels, out_channels))
            in_channels= out_channels

        self.features = nn.Sequential(*layers)
        self.global_avg_pool= nn.AdaptiveAvgPool2d((1,1))
        self.classifier= nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        x=self.global_avg_pool(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)

        return torch.sigmoid(x)