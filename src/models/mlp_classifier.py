from typing import Optional

from lightning import LightningModule
import torch
from torch import nn, optim
from torch.nn.functional import cross_entropy

class MLPClassifier(LightningModule):
    def __init__(self, encoder: nn.Module, *, name: Optional[str]=None):
        super().__init__()
        self._encoder = encoder

    def training_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self._encoder(x)
        loss = cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self._encoder(x)
        loss = cross_entropy(y_pred, y)
        self.log("val_loss", loss)

    def test_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self._encoder(x)
        predicted = torch.max(y_pred, dim=1)
        accuracy = (y_pred == predicted).sum().item()
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer