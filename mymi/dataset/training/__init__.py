import os
import shutil
from typing import List

from mymi import config

from .training_dataset import TrainingDataset
from .training_partition import TrainingPartition

def list() -> List[str]:
    """
    returns: list of raw datasets.
    """
    path = os.path.join(config.directories.datasets, 'training')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def exists(name: str) -> bool:
    """
    returns: if the dataset exists.
    """
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    return os.path.exists(ds_path)

def create(name: str) -> TrainingDataset:
    """
    effect: creates a dataset.
    args:
        name: the name of the dataset.
    """
    # Create root folder.
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    os.makedirs(ds_path)

    return TrainingDataset(name)

def destroy(name: str) -> None:
    """
    effect: destroys a dataset.
    args:
        name: the name of the dataset.
    """
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> None:
    """
    effect: destroys and creates a dataset.
    args:
        name: the name of the dataset.
    """
    destroy(name)
    return create(name)
