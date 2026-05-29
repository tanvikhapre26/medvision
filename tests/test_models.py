import torch
from medvision.models import CustomCNN

def test_custom_cnn_output_shape():
    model = CustomCNN(num_classes=3, num_blocks=4)
    x = torch.rand(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 3)

def test_custom_cnn_output_range():
    model = CustomCNN(num_classes=3)
    x = torch.rand(2, 1, 224, 224)
    out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0

def test_custom_cnn_configurable():
    model = CustomCNN(num_classes=5, num_blocks=3, base_filters=16, dropout=0.5)
    x = torch.rand(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 5)