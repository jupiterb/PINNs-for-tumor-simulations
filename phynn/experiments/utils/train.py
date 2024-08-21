import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
import wandb

from dataclasses import dataclass

from phynn.experiments.models import BaseModel


@dataclass
class RunConfig:
    name: str
    batch_size: int
    epochs: int
    project: str = "physics-learning"


def train(
    model: BaseModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: RunConfig,
) -> None:
    train_dataloader = DataLoader(train_dataset, config.batch_size, True)
    val_dataloader = DataLoader(val_dataset, config.batch_size, False)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")

    logger = WandbLogger(project=config.project, name=config.name, log_model="all")

    try:
        trainer = L.Trainer(
            max_epochs=config.epochs,
            logger=logger,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        wandb.finish()
    except Exception as e:
        print(f"Exception raised: {e}")
        wandb.finish(1)
        raise e
