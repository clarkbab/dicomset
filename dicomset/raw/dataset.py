import os

from .. import config
from ..dataset import Dataset
from ..typing import DatasetID

class RawDataset(Dataset):
    def __init__(
        self,
        id: DatasetID,
        ) -> None:
        self._path = os.path.join(config.directories.datasets, 'raw', str(id))
        if not os.path.exists(self._path):
            raise ValueError(f"No raw dataset '{id}' found at path: {self._path}")
        super().__init__(id)
    