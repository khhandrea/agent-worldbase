from pathlib import Path

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch import Generator
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str | Path):
        super().__init__()
        self._data_dir = data_dir
        self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        MNIST(self._data_dir, train=True, download=True)
        MNIST(self._data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(
                self._data_dir,
                train=True,
                transform=self._transform
            )
            self._mnist_train, self._mnist_val = random_split(
                mnist_full,
                (55000, 5000),
                generator=Generator().manual_seed(42)
            )
        elif stage == "test":
            self._mnist_test = MNIST(
                self._data_dir,
                train=False,
                transform=self._transform
            )
        elif stage == "predict":
            pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._mnist_train, batch_size=32, num_workers=4)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._mnist_val, batch_size=32, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._mnist_test, batch_size=32, num_workers=4)

    def predict_dataloader(self) -> DataLoader:
        pass