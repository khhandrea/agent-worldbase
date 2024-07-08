from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from torch import set_float32_matmul_precision

from networks import MLPEncoder, MLPDecoder
from trainers import AutoEncoder, MLPClassifier
from datasets import MNISTDataModule
from utils import DotDict

if __name__ == "__main__":
    cfg = DotDict({
        "encoder": "mlp_encoder",
        "decoder": "none",
        "model": "mlp_classifier",
        "dataset": "mnist"
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

    if cfg.dataset == "mnist":
        dataset = MNISTDataModule("/data/khhandrea/mnist/")

    seed_everything(42)
    set_float32_matmul_precision("high")

    logger = TensorBoardLogger(
        save_dir="logs/",
        name="hello-rl",
        log_graph=True
    )
    trainer = Trainer(
        devices=[0],
        accelerator="gpu",
    )

    trainer.fit(model, dataset)