import torch
import numpy as np

def minmax_normalize(image):
    """
    Min-max normalizaton: scales pixel value to range [0,1].
    Simple but sensitive to outliers.
    Args:
        image: torch tensor or numpy array
    Returns:
        normalized image as float32 torch tensor 
    """
    if isinstance(image, np.ndarray):
        image= torch.tensor(image, dtype=torch.float32)

    image= image - image.min()
    image= image / (image.max() + 1e-8)

    return image


def zscore_normalize(image):
    """
    Z-score normalization: centers data to mean=0, std=1.
    More robust to outliers. Preferred  for medical imaging.
    Args:
        image: torch tensor or numpy array
    Returns:
        normalized image as float32 torch tensor     
    """

    if isinstance(image, np.ndarray):
        image= torch.tensor(image, dtype=torch.float32)

    mean= image.mean()
    std= image.std()
    image= (image - mean) / (std + 1e-8)

    return image


def normalize(image, modality="xray"):
    """
    Modality-aware normalization.
    Automatically picks the right normalization based on image type.

    Args:
        image    : torch tensor or numpy array
        modality : "xray" or "mri" (default: "xray")
    Returns:
        normalized image as float32 torch tensor
    """
    if modality == "mri":
        return zscore_normalize(image)  # MRI has no standard intensity scale — z-score is always better
    else:
        return minmax_normalize(image)  # X-ray values are more standardised — min-max works well