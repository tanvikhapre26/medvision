<div align="center">

#  medvision

### PyTorch library for medical image classification

*Load. Preprocess. Train. Explain. - in minutes.*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-7%20passed-brightgreen?style=flat-square)]()

---

**medvision** gives you everything you need for medical image classification in one clean package -
DICOM support, modality aware preprocessing, pretrained CNN wrappers, clinical metrics and GradCAM visualization.

No boilerplate. No duct tape. Just results.

</div>

---


## 🚀 Get started in 60 seconds

```bash
git clone https://github.com/tanvikhapre26/medvision.git
cd medvision
pip install -e .
```

```python
from medvision.preprocessing import load_image, normalize, get_transforms
from medvision.models import EfficientNetWrapper
from medvision.training import Trainer
import torch.nn as nn, torch

# 1. load a medical image (DICOM or JPG/PNG)
image= load_image("chest_scan.dcm", size=(224, 224))
image= normalize(image, modality="xray")

# 2. build a pretrained model
model= EfficientNetWrapper(num_classes=3)

# 3. train with one line
trainer= Trainer(model, torch.optim.Adam(model.parameters()), nn.BCELoss())
trainer.fit(train_loader, val_loader, epochs=20)

# 4. explain what the model sees
from medvision.utils import GradCAM
gradcam= GradCAM(model, target_layer=model.model.features[-1])
heatmap= gradcam.generate(image.unsqueeze(0))
gradcam.overlay(image.squeeze().numpy(), heatmap, save_path="gradcam_output.png")
```

---

## 🧠 Models

```python
from medvision.models import CustomCNN, EfficientNetWrapper, DenseNetWrapper, ResNetWrapper

# lightweight configurable CNN from scratch
model= CustomCNN(num_classes=3, num_blocks=4, base_filters=32, dropout=0.3)

# pretrained backbones — freeze/unfreeze layers, replace head automatically
model= EfficientNetWrapper(num_classes=3, unfreeze_layers=2)
model= DenseNetWrapper(num_classes=3)   # great for chest X-rays
model= ResNetWrapper(num_classes=3)     # solid general purpose
```

---

## 🖼️ Preprocessing

```python
from medvision.preprocessing import load_image, normalize, get_transforms

# supports DICOM (.dcm), JPG, PNG
image= load_image("scan.dcm")              # returns torch tensor (1, 224, 224)

# modality-aware normalization
image= normalize(image, modality="xray")   # min-max for X-ray
image= normalize(image, modality="mri")    # z-score for MRI

# augmentation pipelines built for medical imaging
# X-ray: CLAHE contrast, small rotation (no horizontal flip — left/right matters clinically)
# MRI: elastic transform, gaussian noise, rotation
transform= get_transforms(modality="xray", train=True)
```

---

## 📊 Training

```python
from medvision.training import Trainer, EarlyStopping, ModelCheckpoint

callbacks= [
    EarlyStopping(patience=5),
    ModelCheckpoint(path="best_model.pth"),
]

trainer= Trainer(model, optimizer, criterion, device="cuda", callbacks=callbacks)
trainer.fit(train_loader, val_loader, epochs=30)
trainer.save("final_model.pth")
```

---

## 📈 Clinical Metrics

```python
from medvision.training import compute_auc, compute_f1, compute_sensitivity, compute_specificity

# metrics that matter in medical imaging
print("AUC:        ", compute_auc(y_true, y_pred))
print("F1:         ", compute_f1(y_true, y_pred))
print("Sensitivity:", compute_sensitivity(y_true, y_pred))  # catching true positives
print("Specificity:", compute_specificity(y_true, y_pred))  # avoiding false alarms
```

---

## 🔥 GradCAM — see what your model sees

GradCAM highlights which regions of the image drove the model's decision.
Critical for medical imaging — verify your model looks at the right anatomy, not scanner artifacts.

```python
from medvision.utils import GradCAM

gradcam= GradCAM(model, target_layer=model.model.features[-1])
heatmap= gradcam.generate(image.unsqueeze(0), class_idx=1)
overlay= gradcam.overlay(image.squeeze().numpy(), heatmap, alpha=0.4, save_path="gradcam.png")
```

---

## 📦 Package structure

```
medvision/
├── preprocessing/    load_image · normalize · get_transforms · MedicalDataset
├── models/           CustomCNN · EfficientNetWrapper · DenseNetWrapper · ResNetWrapper
├── training/         Trainer · EarlyStopping · ModelCheckpoint · metrics
└── utils/            GradCAM · plot_training_curves · plot_confusion_matrix · plot_roc_curve
```

---

## 🧪 Run tests

```bash
pytest tests/ -v
```

```
7 passed in 5.52s ✅
```

---

<div align="center">

Built by **Tanvi Khapre**

</div>
