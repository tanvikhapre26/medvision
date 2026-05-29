import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_xray_transforms (train= True):
    """
    Augmentation pipeline for X-ray images.
    train=True  : full augmentation for training
    train=False : no augmentation for validation/testing
    
    Note: No horizontal flip — left/right orientation is
    clinically significant in X-rays.
    """
    if train:
        return A.Compose([
            A.CLAHE(clip_limit= 2.0, tile_grid_size= (8,8), p= 0.5),
            A.Rotate(limit= 10, p= 0.5 ),
            A.RandomBrightnessContrast (p= 0.3),
        ])
    else:
        return A.Compose([])
    
def get_mri_transforms (train= True):
    """
    Augmentation pipeline for MRI images.
    train=True  : full augmentation for training
    train=False : no augmentation for validation/testing

    Note: Elastic transform mimics natural tissue deformation
    seen across different patients and scanners.
    """
    if train: 
        return A.Compose([
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ])
    else: 
        return A.Compose([])
    
def get_transforms(modality="xray",train=True):
    """
    Returns the appropriate augmentation pipeline based on modality.

    Args:
        modality : "xray" or "mri" (default: "xray")
        train    : True for training augmentation, False for validation
    Returns:
        albumentations Compose pipeline
    
    Usage:
        transform = get_transforms(modality="mri", train=True)
    """
    if modality=="mri":
        return get_mri_transforms(train=train)
    else:
        return get_xray_transforms(train=train)