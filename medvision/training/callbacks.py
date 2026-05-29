import torch

class EarlyStopping:
    """
    Stops training if validation loss doesn't improve for 'patience' epochs.

    Usage:
        early_stop = EarlyStopping(patience=5)
        early_stop(val_loss)
        if early_stop.stop:
            break
    """
    def __init__(self, patience=5):
        self.patience= patience
        self.counter= 0
        self.best_loss= None
        self.stop= False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss= val_loss   # first epoch — set baseline

        elif val_loss < self.best_loss:
            self.best_loss= val_loss   # improvement — reset counter
            self.counter= 0

        else:
            self.counter +=1
            if self.counter  >= self.patience:
                self.stop = True
                print(f"Early stopping triggered after {self.patience} epochs of no improvement.")

class ModelCheckpoint:
    """
    Saves the model when validation AUC improves.

    Usage:
        checkpoint = ModelCheckpoint(path="best_model.pth")
        checkpoint(val_auc, model)
    """
    def __init__(self, path="best_model.pth"):
        self.path= path
        self.best_auc= None

    def __call__(self, val_auc, model):
        if self.best_auc is None or val_auc > self.best_auc:
            self.best_auc= val_auc
            torch.save(model.state_dict(), self.path)
            print(f"Model saved, val AUc improved to {val_auc:.4f}")

class LRScheduler:
    """
    Wraps a PyTorch learning rate scheduler and steps it each epoch.

    Usage:
        scheduler = LRScheduler(torch.optim.lr_scheduler.StepLR(optimizer, step_size=5))
        scheduler(epoch)
    """
    def __init__(self, scheduler):
        self.scheduler= scheduler

    def __call__(self, epoch):
        self.scheduler.step()
        lr= self.scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} - learning rate {lr:.6f}")