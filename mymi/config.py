from collections import namedtuple
import os
import pandas as pd
from typing import List, Optional

from mymi import logging

class Directories:
    @property
    def cache(self):
        return os.path.join(self.root, 'cache')

    @property
    def models(self):
        return os.path.join(self.root, 'models')

    @property
    def datasets(self):
        return os.path.join(self.root, 'datasets')

    @property
    def files(self):
        return os.path.join(self.root, 'files')
    
    @property
    def evaluation(self):
        return os.path.join(self.root, 'evaluation')

    @property
    def root(self):
        return os.environ['MYMI_DATA']

    @property
    def temp(self):
        return os.path.join(self.root, 'tmp')

    @property
    def tensorboard(self):
        return os.path.join(self.root, 'reporting', 'tensorboard')

    @property
    def wandb(self):
        return os.path.join(self.root, 'reporting')

class Formatting:
    @property
    def metrics(self):
        return '.6f'

    @property
    def sample_digits(self):
        return 5

directories = Directories()
formatting = Formatting()

def environ(name: str) -> Optional[str]:
    if name in os.environ:
        return os.environ[name]
    else:
        return None

def save_csv(
    data: pd.DataFrame,
    *path: List[str],
    index: bool = False,
    overwrite: bool = False) -> None:
    filepath = os.path.join(directories.files, *path)
    dirpath = os.path.dirname(filepath)
    if os.path.exists(filepath):
        if overwrite:
            os.makedirs(dirpath, exist_ok=True)
            data.to_csv(filepath, index=index)
        else:
            logging.error(f"File '{filepath}' already exists, use overwrite=True.")
    else:
        os.makedirs(dirpath, exist_ok=True)
        data.to_csv(filepath, index=index)

def load_csv(*path: List[str]) -> Optional[pd.DataFrame]:
    filepath = os.path.join(directories.files, *path)
    return pd.read_csv(filepath)
