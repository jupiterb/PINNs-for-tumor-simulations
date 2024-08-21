import os

from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

from phynn.data import FactoryBasedDataset, HDF5DataInterfaceFactory
from phynn.experiments.utils.train import train, RunConfig
from phynn.experiments.utils.device import training_device
from phynn.models import BaseModel, OptimizerParams


@dataclass
class StageConfig:
    optimizer_params: OptimizerParams


@dataclass
class RunStageConfig(StageConfig):
    train_data_path: Path
    val_data_path: Path
    run_config: RunConfig


@dataclass
class CheckpointStageConfig(StageConfig):
    checkpoint_path: Path


ModelType = TypeVar("ModelType", bound=BaseModel)
DatasetType = TypeVar("DatasetType", bound=FactoryBasedDataset)


def run_stage(
    model_cls: Type[ModelType],
    dataset_cls: Type[DatasetType],
    config: StageConfig,
    **model_params
) -> ModelType | None:
    if isinstance(config, RunStageConfig):
        model = model_cls(**model_params)
        run_training(config, model, dataset_cls)
        return model
    elif isinstance(config, CheckpointStageConfig):
        return model_cls.load_from_checkpoint(config.checkpoint_path, **model_params)


def run_training(
    config: RunStageConfig,
    model: BaseModel,
    dataset_cls: Type[DatasetType],
) -> None:
    train_ds = get_dataset(config.train_data_path, dataset_cls)
    val_ds = get_dataset(config.val_data_path, dataset_cls)
    train(model, train_ds, val_ds, config.run_config)


def get_dataset(path: os.PathLike, dataset_cls: Type[DatasetType]) -> DatasetType:
    factory = HDF5DataInterfaceFactory(path, training_device)
    return dataset_cls(factory)
