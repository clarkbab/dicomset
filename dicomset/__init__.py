from typing import List

from .dataset import Dataset
from .dicom import DicomDataset
from .dicom import list as list_dicom
from .nifti import NiftiDataset
from .nifti import list as list_nifti
from .raw import RawDataset
from .raw import list as list_raw
from .training import TrainingDataset
from .training import list as list_training
from .typing import DatasetID, DatasetType

def get(
    name: str,
    type: DatasetType,
    **kwargs,
    ) -> Dataset:
    if type == 'dicom':
        return DicomDataset(name, **kwargs)
    elif type == 'nifti':
        return NiftiDataset(name, **kwargs)
    elif type == 'raw':
        return RawDataset(name, **kwargs)
    elif type == 'training':
        return TrainingDataset(name, **kwargs)
    else:
        raise ValueError(f"Dataset type '{type}' not found.")

def list(
    type: DatasetType,
    ) -> List[DatasetID]:
    if type == 'dicom':
        return list_dicom()
    elif type == 'nifti':
        return list_nifti()
    elif type == 'raw':
        return list_raw()
    elif type == 'training':
        return list_training()
    else:
        raise ValueError(f"Dataset type '{type}' not found.")
