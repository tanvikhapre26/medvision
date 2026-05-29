import torch
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix)


def compute_auc(y_true, y_pred):
    """
    Compute AUC-ROC score.
    For multi-label, returns macro average across all classes.

    Args:
        y_true : numpy array of true labels
        y_pred : numpy array of predicted probabilities
    Returns:
        AUC score as float
    """
    return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

def compute_f1(y_true, y_pred, threshold=0.5):
    """
    Compute F1 score for multi-label classification.
    Threshold converts probabilities to binary predictions.

    Args:
        y_true      : numpy array of true labels
        y_pred      : numpy array of predicted probabilities
        threshold   : cutoff to convert probability to 0/1 (default 0.5)
    Returns:
        F1 score as float
    """
    y_pred_binary= (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_binary, average="macro", zero_division=0)

def compute_sensitivity(y_true, y_pred, threshold=0.5):
    """
    Compute sensitivity (recall) — how many actual positives did we catch?
    sensitivity = TP / (TP + FN)
    Critical in medical imaging — missing a disease is dangerous.

    Args:
        y_true     : numpy array of true labels
        y_pred     : numpy array of predicted probabilities
        threshold  : cutoff to convert probability to 0/1 (default 0.5)
    Returns:
        sensitivity as float
    """
    y_pred_binary= (y_pred>= threshold).astype(int)
    cm= confusion_matrix (y_true.flatten(), y_pred_binary.flatten())
    tp= cm[1,1]
    fn= cm[1,0]
    return tp / (tp + fn + 1e-8)

def compute_specificity(y_true, y_pred, threshold=0.5):
    """
    Compute specificity — how many actual negatives did we correctly identify?
    specificity = TN / (TN + FP)
    Important to avoid unnecessary treatment of healthy patients.

    Args:
        y_true     : numpy array of true labels
        y_pred     : numpy array of predicted probabilities
        threshold  : cutoff to convert probability to 0/1 (default 0.5)
    Returns:
        specificity as float
    """
    y_pred_binary= (y_pred>= threshold).astype(int)
    cm= confusion_matrix(y_true.flatten(), y_pred_binary.flatten())
    tn= cm[0,0]
    fp= cm[0,1]
    return tn/(tn + fp + 1e-8)

def compute_confusion_matrix(y_true, y_pred, threshold=0.5):
     """
    Compute confusion matrix for binary or multi-label classification.

    Args:
        y_true     : numpy array of true labels
        y_pred     : numpy array of predicted probabilities
        threshold  : cutoff to convert probability to 0/1 (default 0.5)
    Returns:
        confusion matrix as numpy array
    """
     y_pred_binary= (y_pred>= threshold).astype(int)
     return confusion_matrix(y_true.flatten(), y_pred_binary.flatten())