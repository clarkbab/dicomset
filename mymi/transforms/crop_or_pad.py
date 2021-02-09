import hashlib
import json
import math
import numpy as np

class CropOrPad:
    def __init__(self, resolution, fill=0):
        """
        args:
            resolution: an (x, y) tuple of the new resolution.
        kwargs:
            fill: value to use for new pixels.
            p: the probability that the transform is applied.
        """
        self.resolution = resolution
        self.fill = fill

    def __call__(self, data, binary=False, info=None):
        """
        returns: transformed data.
        args:
            the data to transform.
        kwargs:
            binary: is the data binary data.
            info: extra info.
        """
        # Preserve data type.
        data_type = data.dtype

        # Determine which dimensions to reshape.
        resolution = [r if r is not None else d for r, d in zip(self.resolution, data.shape)]

        # Create placeholder array.
        fill = 0 if binary else self.fill
        new_data = np.full(shape=resolution, fill_value=fill, dtype=data.dtype)

        # Find data centres as we will perform centred cropping and padding.
        data_centre = (np.array(data.shape) - 1) / 2
        new_data_centre = (np.array(new_data.shape) - 1) / 2

        # Find the write range.
        write_shape = np.minimum(new_data.shape, data.shape)
        write_lower_bound = np.array(list(map(math.ceil, new_data_centre - write_shape / 2)))
        write_range = [slice(l, l + r) for l, r in zip(write_lower_bound, write_shape)]

        # Find the read range.
        read_lower_bound = np.array(list(map(math.ceil, data_centre - write_shape / 2)))
        read_range = [slice(l, l + r) for l, r in zip(read_lower_bound, write_shape)]

        # Add data to placeholder.
        new_data[tuple(write_range)] = data[tuple(read_range)]

        # Reset data type.
        new_data = new_data.astype(data_type)

        return new_data

    def deterministic(self):
        """
        returns: a deterministic function with same signature as '__call__'.
        """
        # No randomness, just return function identical to '__call__'.
        def fn(data, binary=False, info=None):
            return self.__call__(data, binary=binary, info=info)

        return fn

    def cache_key(self):
        """
        returns: an ID that is unique based upon transform parameters.
        """
        params = {
            'fill': self.fill,
            'resolution': self.resolution
        }
        return hashlib.sha1(json.dumps(params).encode('utf-8')).hexdigest()