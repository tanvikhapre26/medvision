import pytest 
import torch
import numpy as np
from medvision.preprocessing import load_image, normalize, get_transforms
from medvision.preprocessing.loader import MedicalDataset

def test_minmax_normalize():
    from medvision.preprocessing.normalization import minmax_normalize
    img= torch.tensor([0.0, 128.0, 255.0])
    out= minmax_normalize(img)
    assert out.min() >= 0.0
    assert out.max() <=1.0

def test_zscore_normalize():
    from medvision.preprocessing.normalization import zscore_normalize
    img= torch.rand([1, 224, 224])
    out= zscore_normalize(img)
    assert abs(out.mean().item()) < 0.01
    assert abs(out.std().item() - 1.0) < 0.01

def test_normalize_modality():
    img = torch.rand(1, 224, 224)
    xray_out = normalize(img, modality="xray")
    mri_out = normalize(img, modality="mri")
    assert xray_out.shape == img.shape
    assert mri_out.shape == img.shape

def test_medical_dataset():
    import numpy as np
    from PIL import Image
    import tempfile
    import os
    img= Image.fromarray(np.uint8(np.random.rand(224, 224) * 255))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        path= f.name
    
    dataset= MedicalDataset([path], [0])
    image, label= dataset[0]

    assert image.shape== (1, 224, 224)
    assert label.item()== 0.0

    os.unlink(path)