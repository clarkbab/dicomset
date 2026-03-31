import os
import shutil
from typing import List

from .. import config
from ..typing import DatasetID
from ..utils.misc import with_makeitso
from .dataset import RawDataset

def create(
    id: DatasetID,
    recreate: bool = False,
    ) -> RawDataset:
    ds_path = os.path.join(config.directories.datasets, 'raw', id)
    if os.path.exists(ds_path):
        if recreate:
            shutil.rmtree(ds_path)
    os.makedirs(ds_path, exist_ok=True)
    return RawDataset(id)

def destroy(
    id: DatasetID,
    makeitso: bool = True,
    ) -> None:
    ds_path = os.path.join(config.directories.datasets, 'raw', id)
    if os.path.exists(ds_path):
        with_makeitso(makeitso, lambda: shutil.rmtree(ds_path), f"Destroying raw dataset '{id}' at {ds_path}.")

def exists(id: DatasetID) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'raw', id)
    return os.path.exists(ds_path)

def load(id: DatasetID) -> RawDataset:
    ds_path = os.path.join(config.directories.datasets, 'raw', id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Raw dataset '{id}' not found at {ds_path}.")
    return RawDataset(id)
    
def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'raw')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []
