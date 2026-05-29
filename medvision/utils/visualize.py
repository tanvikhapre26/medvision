import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc,  confusion_matrix
import seaborn as sns

def plot_training_curves(history):
    """
    Plot training and validation loss + validation AUC over epochs.

    Args:
        history : dict with keys 'train_loss', 'val_loss', 'val_auc'
                  (returned by trainer.history)
    """
    epochs= range(1, len(history["train_loss"])+1)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # AUC plot
    ax2.plot(epochs, history["val_auc"], label="Val AUC", color="green")
    ax2.set_title("AUC over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names= None, threshold=0.5):
    """
    Plot confusion matrix as a heatmap.

    Args:
        y_true      : numpy array of true labels
        y_pred      : numpy array of predicted probabilities
        class_names : list of class names for axis labels
        threshold   : cutoff to convert probabilities to 0/1
    """
    y_pred_binary= (y_pred>= threshold).astype(int)
    cm= confusion_matrix(y_true, y_pred_binary)

    fig, ax= plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()
    return fig


def plot_roc_curve(y_true, y_pred):
    """
    Plot ROC curve with AUC score.

    Args:
        y_true : numpy array of true labels
        y_pred : numpy array of predicted probabilities
    """
    fpr, tpr, _= roc_curve(y_true, y_pred)
    auc_score= auc(fpr, tpr)

    fig, ax= plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC= {auc_score:.4f}", color="blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    plt.tight_layout()
    plt.show()
    return fig