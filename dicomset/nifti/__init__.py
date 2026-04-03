import os
import shutil
from typing import List

from .. import config
from ..typing import DatasetID
from ..utils.misc import with_makeitso
from .dataset import NiftiDataset
from .patient import NiftiPatient
from .series import NiftiCtSeries, NiftiDoseSeries, NiftiImageSeries, NiftiLandmarksSeries, NiftiMrSeries, NiftiRegionsSeries
from .study import NiftiStudy

def create(
    id: DatasetID,
    recreate: bool = False,
    ) -> NiftiDataset:
    ds_path = os.path.join(config.directories.datasets, 'nifti', id)
    if os.path.exists(ds_path):
        if recreate:
            shutil.rmtree(ds_path)
    os.makedirs(ds_path, exist_ok=True)
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

def load(id: DatasetID) -> NiftiDataset:
    ds_path = os.path.join(config.directories.datasets, 'nifti', id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Nifti dataset '{id}' not found at {ds_path}.")
    return NiftiDataset(id)
    
def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'nifti')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []
