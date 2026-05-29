import pydicom
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

def load_image (path, size=(224,224)):
    """
    Load a single image from afile path.
    Supports DICOM (.dcm) and standard formats (JPG, PNG).
    Returns a normalized float32 PyTorch of shape (1, H, W).
    """
    path=Path(path)
    if path.suffix.lower() == ".dcm":
        ds= pydicom.dcmread(str(path))
        image= ds.pixel_array.astype(np.float32)  # pixel_array gives us the raw image as a numpy array

        # normalizing the dICOM as pixel value goes upto 4095 ( normalize to 0-225).
        image= image -image.min()
        image= image / (image.max() + 1e-8)* 255.0

    else:
        # For JPG/PNG — open with PIL, convert to grayscale and 'L'  for grayscale with values 0-225.
        image= Image.open(str(path)).convert("L")
        image= np.array (image, dtype= np.float32)

    image_pil= Image.fromarray(image).resize(size, Image.BILINEAR)
    image= np.array(image_pil, dtype=np.float32)
    image= image /255.0
    image= np.expand_dims(image, axis=0) #medical images are grayscale so we get (1, 224, 224)

    return torch.tensor(image, dtype= torch.float32)


class MedicalDataset(Dataset):
    """
    PyTorch Dataset for medical images.
    Handles a list of image paths and their corresponding labels.
    Supports both DICOM and standard image formats.

    Usage:
        dataset = MedicalDataset(paths, labels, modality="xray")
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, image_paths, labels, size=(224,224), modality="xray", transform=None):
        """
        Args:
            image_paths : list of file paths to images
            labels      : list of labels (int for single label, list for multi-label)
            size        : target image size as (H, W), default 224x224
            modality    : "xray" or "mri" — used later by augmentation/normalisation
            transform   : optional albumentations transform pipeline
        """
        self.image_paths= image_paths
        self.labels= labels
        self.size= size
        self.modality= modality
        self.transform= transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image= load_image(self.image_paths[idx], size= self.size)
        label= self.labels[idx]


        if self.transform is not None:
            # albumentations expects numpy HxW format, not tensor
            # so we convert back, transform, then back to tensor
            image_np= image.squeeze(0).numpy()
            augmented= self.transform(image=image_np)
            image_np= augmented["image"]
            image=  torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)

        label= torch.tensor(label, dtype=torch.float32)

        return image, label
        