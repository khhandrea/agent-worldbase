from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from networks import MLPEncoder, MLPDecoder
from trainers import AutoEncoder, MLPClassifier
from utils import DotDict

if __name__ == "__main__":
    cfg = DotDict({
        "encoder": "mlp_encoder",
        "decoder": "none",
        "model": "mlp_classifier"
    })

    if cfg.encoder == "mlp_encoder":
        encoder = MLPEncoder()
    else:
        encoder = None

    if cfg.decoder == "mlp_decoder":
        decoder == MLPDecoder()
    else:
        decoder = None

    if cfg.model == "mlp_classifier":
        model = MLPClassifier(encoder)
    elif cfg.model == "autoencoder":
        model = AutoEncoder(encoder, decoder)

    seed_everything(42)
    set_float32_matmul_precision("high")

    train_dataset = MNIST("/data/khhandrea/mnist", download=True, train=True, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4)

    logger = TensorBoardLogger(
        save_dir="logs/",
        name="hello-rl",
        log_graph=True
    )
    trainer = Trainer(
        max_epochs=10,
        devices=[4],
        accelerator="gpu",
    )

    trainer.fit(model, train_dataloaders=train_dataloader)