import inspect
import numpy as np
import os

from mymi import cache

FILENAME_NUM_DIGITS = 5

class ProcessedDataset:
    ###
    # Subclasses must implement.
    ###

    @classmethod
    def data_dir(cls):
        raise NotImplementedError("Method 'data_dir' not implemented in subclass.")

    ###
    # Basic queries.
    ###

    @classmethod
    def input(cls, folder, sample_idx):
        """
        returns: the input data for sample i.
        args:
            folder: 'train', 'test' or 'validate'.
            sample_idx: the sample index to load.
        """
        # Load the input data.
        filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-input"
        input_path = os.path.join(cls.data_dir(), folder, filename)
        f = open(input_path, 'rb')
        input = np.load(f)

        return input

    @classmethod
    def label(cls, folder, sample_idx):
        """
        returns: the label data for sample i.
        args:
            folder: 'train', 'test' or 'validate'.
            sample_idx: the sample index to load.
        """
        # Load the label data.
        filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-label"
        label_path = os.path.join(cls.data_dir(), folder, filename)
        f = open(label_path, 'rb')
        label = np.load(f)

        return label

    @classmethod
    def sample(cls, folder, sample_idx):
        """
        returns: the (input, label) pair for the given sample index.
        args:
            folder: 'train', 'test' or 'validate'.
            sample_idx: the sample index to load.
        """
        return cls.input(folder, sample_idx), cls.label(folder, sample_idx)

    # TODO: Move to subclass if necessary.
    @classmethod
    def class_frequencies(cls, folder):
        """
        returns: the frequencies for each class.
        args:
            folder: 'train', 'test', or 'validate'.
        """
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'args': {
                'folder': folder
            }
        }
        result = cache.read(params, 'array')
        if result is not None:
            return tuple(result)

        # Get number of samples.
        folder_path = os.path.join(cls.data_dir(), folder)
        num_samples = int(len(os.listdir(folder_path)) / 2)

        # Count background and foreground frequencies.
        num_back, num_fore = 0, 0
        for i in range(num_samples):
            # Load label.
            label_data = cls.label(folder, i)

            # Add values.
            shape = label_data.shape
            sum = label_data.sum()
            num_back += shape[0] * shape[1] * shape[2] - sum
            num_fore += sum

        # Create frequencies tuple.
        freqs = (num_back, num_fore)

        # Write data to cache.
        cache.write(params, freqs, 'array')

        return freqs