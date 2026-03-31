import os
import shutil
from typing import List

from .. import config
from ..typing import DatasetID
from ..utils.misc import with_makeitso
from .dataset import NiftiDataset

def create(id: DatasetID) -> NiftiDataset:
    ds_path = os.path.join(config.directories.datasets, 'nifti', id)
    if os.path.exists(ds_path):
        raise FileExistsError(f"Dataset '{id}' already exists at {ds_path}.")
    os.makedirs(ds_path)
    return NiftiDataset(id)

def destroy(
    id: DatasetID,
    makeitso: bool = True,
    ) -> None:
    ds_path = os.path.join(config.directories.datasets, 'nifti', id)
    if os.path.exists(ds_path):
        with_makeitso(makeitso, lambda: shutil.rmtree(ds_path), f"Destroying nifti dataset '{id}' at {ds_path}.")

def exists(id: DatasetID) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'nifti', id)
    return os.path.exists(ds_path)
    
def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'nifti')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []

def recreate(
    id: DatasetID,
    makeitso: bool = True,
    ) -> NiftiDataset:
    destroy(id, makeitso=makeitso)
    if makeitso:
        return create(id)
    else:
        if exists(id):
            return NiftiDataset(id)
        else:
            # Creating is fine with makeitso=False.
            return create(id)
