import os

DATA_ENV = 'DS_DATA'

class Directories:
    @property
    def datasets(self):
        filepath = os.path.join(self.data, 'datasets')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def files(self):
        filepath = os.path.join(self.data, 'files')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def data(self):
        if DATA_ENV not in os.environ:
            raise ValueError(f"Must set env var '{DATA_ENV}' for DicomSet.")
        return os.environ[DATA_ENV]

directories = Directories()
