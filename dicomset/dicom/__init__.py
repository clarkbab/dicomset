import os
import shutil
from typing import List

from .. import config
from ..typing import DatasetID
from ..utils.misc import with_makeitso
from .dataset import DicomDataset
from .patient import DicomPatient
from .series import DicomCtSeries, DicomMrSeries, DicomRtDoseSeries, DicomRtPlanSeries, DicomRtStructSeries, DicomSeries
from .study import DicomStudy

def create(
    id: DatasetID,
    recreate: bool = False,
    ) -> DicomDataset:
    ds_path = os.path.join(config.directories.datasets, 'dicom', id)
    if os.path.exists(ds_path):
        if recreate:
            shutil.rmtree(ds_path)
    os.makedirs(ds_path, exist_ok=True)
    return DicomDataset(id)

def destroy(
    id: DatasetID,
    makeitso: bool = True,
    ) -> None:
    ds_path = os.path.join(config.directories.datasets, 'dicom', id)
    if os.path.exists(ds_path):
        with_makeitso(makeitso, lambda: shutil.rmtree(ds_path), f"Destroying dicom dataset '{id}' at {ds_path}.")

def exists(id: DatasetID) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'dicom', id)
    return os.path.exists(ds_path)

def load(id: DatasetID) -> DicomDataset:
    ds_path = os.path.join(config.directories.datasets, 'dicom', id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dicom dataset '{id}' not found at {ds_path}.")
    return DicomDataset(id)
    
def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'dicom')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []
