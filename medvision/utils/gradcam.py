import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# GradCAM = Gradient-weighted Class Activation Mapping
class GradCAM:
    """
    GradCAM visualization for CNN models.
    Shows which regions of the image the model focused on.

    Usage:
        gradcam = GradCAM(model, target_layer=model.features[-1])
        heatmap = gradcam.generate(image)
    """
    def __init__(self, model, target_layer):
        self.model= model
        self.target_layer= target_layer
        self.gradients= None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activations(self, module, input, output):
        self.activations= output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients= grad_output[0].detach()


    def generate(self, image, class_idx= None):
        """
        Generate GradCAM heatmap for a given image.

        Args:
            image     : torch tensor of shape (1, C, H, W)
            class_idx : target class index (None = uses predicted class)
        Returns:
            heatmap as numpy array (H, W), values 0-1
        """
        self.model.eval()
        output= self.model(image)
        if class_idx is None:
            class_idx= output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights= self.gradients.mean(dim=(2,3), keepdim=True)
        heatmap= (weights * self.activations).sum(dim=1).squeeze()
        heatmap= torch.relu(heatmap).cpu().numpy()

        heatmap= heatmap - heatmap.min()
        heatmap= heatmap / (heatmap.max() + 1e-8)
        return heatmap
    
    
    def overlay(self, image, heatmap, alpha=0.5, save_path=None):
        """
        Overlay GradCAM heatmap on original image.

        Args:
            image     : original image as numpy array (H, W)
            heatmap   : heatmap from generate() method
            alpha     : heatmap transparency (default 0.4)
            save_path : if provided, saves the output image to this path
        Returns:
            overlaid image as numpy array
        """
        heatmap_resized= cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored= cv2.applyColorMap(
            np.uint8(255*heatmap_resized),  cv2.COLORMAP_JET
        )

        if len(image.shape) ==2:
            image_rgb= cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_GRAY2RGB)

        else:
            image_rgb= np.uint8(image * 255)
        
        overlaid= cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)

        if save_path:
            cv2.imwrite(save_path, overlaid)
            print(f"GradCAM saved to {save_path}")

        return overlaid