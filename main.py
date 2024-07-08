import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.pytorch import seed_everything
from torch import set_float32_matmul_precision

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(42)
    set_float32_matmul_precision("high")

    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)
    trainer = instantiate(cfg.trainer, logger=cfg.logger)

    trainer.fit(model, dataset)

if __name__ == "__main__":
    main()