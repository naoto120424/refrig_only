import numpy as np
import torch
import mlflow
import os


class EarlyStopping:

    def __init__(self, path, patience=7, verbose=False, delta=1e-3, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(path, "best_model.pth")
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            mlflow.log_metric(f"best epoch num", epoch, step=epoch)
        elif score > self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            mlflow.log_metric(f"best epoch num", epoch - self.counter, step=epoch)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            mlflow.log_metric(f"best epoch num", epoch, step=epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f"This is the best model. Save to best_model.pth")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
