import torch 
import numpy as np
from .metrics import compute_auc, compute_f1

class Trainer:
    """
    Training loop for medical image classification models.
    Handles training, validation, evaluation, and model saving.

    Usage:
        trainer = Trainer(model, optimizer, criterion, device="cuda")
        trainer.fit(train_loader, val_loader, epochs=20)
    """
    def __init__(self, model, optimizer, criterion, device="cpu", callbacks=None):
        self.model= model
        self.optimizer= optimizer
        self.criterion= criterion
        self.device= device
        self.callbacks= callbacks or []

        self.model.to(self.device)
        self.history= {
            "train_loss": [],
            "val_loss": [],
            "val_auc": []
        }

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss=0

        for images, labels in train_loader:
            images= images.to(self.device)
            labels= labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs= self.model(images)
            loss= self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)
    
    def _val_epoch(self, val_loader):
        self.model.eval()
        total_loss=0
        all_preds= []
        all_labels= []

        with  torch.no_grad():
            for images, labels in val_loader:
                images= images.to(self.device)
                labels= labels.to(self.device)

                outputs= self.model(images)
                loss= self.criterion(outputs, labels)
                total_loss += loss.item()

                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds= np.concatenate(all_preds)
        all_labels= np.concatenate(all_labels)

        val_loss= total_loss/ len(val_loader)
        val_auc= compute_auc(all_labels, all_preds)

        return val_loss, val_auc
    
    def fit(self, train_loader, val_loader, epochs=20):
        for epoch in range (1, epochs+1):
            train_loss= self._train_epoch(train_loader)
            val_loss, val_auc= self._val_epoch(val_loader)

            # store in history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_auc: {val_auc:.4f}")

            for callback in self.callbacks:
                if  hasattr(callback, 'stop'):
                    callback(val_loss)
                    if callback.stop:
                        print("Trainig stopped early.")
                        return
                else:
                        callback(epoch)

    def evaluate(self, test_loader):
        val_loss, val_auc =self._val_epoch(test_loader)
        val_f1 = compute_f1(
            np.concatenate([l.cpu().numpy() for _, l in test_loader]),
            np.concatenate([self.model(i.to(self.device)).detach().cpu().numpy() for i, _ in test_loader])
        )
        print(f"Test_loss: {val_loss:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        return {"loss": val_loss, "auc": val_auc, "f1": val_f1}
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved to {path}")

    def load(self,path):
        self.model.load_state_dict(torch.load(path, map_location= self.device))
        print("Model loaded from {path}")