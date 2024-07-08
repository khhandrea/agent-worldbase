from lightning import LightningModule
from torch import nn, optim
from torch.nn.functional import mse_loss

class AutoEncoder(LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def training_step(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self._encoder(x)
        x_pred = self._decoder(z)
        loss = mse_loss(x_pred, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
