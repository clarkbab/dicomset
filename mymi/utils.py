from colorlog import ColoredFormatter
import gc
import logging
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from typing import *

def escape_latex(text: str) -> str:
    """
    returns: a string with escaped latex special characters.
    args:
        text: the string to escape.
    """
    # Provide map for special characters.
    char_map = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(char_map.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: char_map[match.group()], text)

def pretty_size(size):
    assert isinstance(size, torch.Size)
    return ' x '.join(map(str, size))

def dump_tensors(gpu_only=True):
    """
    Prints list of Tensors that are tracked by the garbage collector.
    """
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(f"{type(obj).__name__}:{' GPU' if obj.is_cuda else ''}{' pinned' if obj.is_pinned else ''} {pretty_size(obj.size())}")
                    total_size += obj.numel()
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        print(f"{type(obj).__name__} -> {type(obj.data).__name__}:{' GPU' if obj.is_cuda else ''}{' pinned' if obj.data.is_pinned else ''}{' grad' if obj.requires_grad else ''}{' volatile' if obj.volatile else ''} {pretty_size(obj.data.size())}")
                        total_size += obj.data.numel()
        except Exception as e:
            pass
    print(f"Total size: {total_size}")

def configure_device():
    num_devices = torch.cuda.device_count()
    logging.info(f"Found {num_devices} device/s:")
    for i in range(num_devices):
        logging.info(f"\t{torch.cuda.get_device_name(i)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logging.info('Running on GPU!')
    else:
        logging.info('Using CPU.')
    return device

def configure_logging(log_level):
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {log_level}.")
    log_format = "%(log_color)s%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ColoredFormatter(log_format, date_format)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logging.basicConfig(handlers=[stream], level=log_level)

def binary_perimeter(mask):
    mask_perimeter = torch.zeros_like(mask, dtype=bool)
    b_dim, x_dim, y_dim = mask.shape
    for b in range(b_dim):
        for i in range(x_dim):
            for j in range(y_dim):
                # Check if edge pixel.
                if (mask[b, i, j] == 1 and 
                    ((i == 0 or i == x_dim - 1) or
                    (j == 0 or j == y_dim - 1) or
                    i != 0 and mask[b, i - 1, j] == 0 or 
                    i != x_dim - 1 and mask[b, i + 1, j] == 0 or
                    j != 0 and mask[b, i, j - 1] == 0 or
                    j != y_dim - 1 and mask[b, i, j + 1] == 0)):
                    mask_perimeter[b, i, j] = 1
    return mask_perimeter

def get_batch_centroids(label_batch, plane):
    """
    returns: the centroid location of the label along the plane axis, for each
        image in the batch.
    args:
        label_batch: the batch of labels.
        plane: the plane along which to find the centroid.
    """
    assert plane in ('axial', 'coronal', 'sagittal')

    # Move data to CPU.
    label_batch = label_batch.cpu()

    # Determine axes to sum over.
    if plane == 'axial':
        axes = (0, 1)
    elif plane == 'coronal':
        axes = (0, 2)
    elif plane == 'sagittal':
        axes = (1, 2)

    centroids = np.array([], dtype=np.int)

    # Loop through batch and get centroid for each label.
    for label_i in label_batch:
        # Get weighting along 'plane' axis.
        weights = label_i.sum(axes)

        # Get average weighted sum.
        indices = np.arange(len(weights))
        avg_weighted_sum = (weights * indices).sum() /  weights.sum()

        # Get centroid index.
        centroid = np.round(avg_weighted_sum).long()
        centroids = np.append(centroids, centroid)

    return centroids

def filter_on_num_pats(num_pats: int) -> Callable[[str], bool]:
    """
    returns: a function to filter patients by number of patients allowed.
    args:
        num_pats: the number of patients to keep.
    """
    def fn(id):
        if num_pats == 'all' or fn.num_included < num_pats:
            fn.num_included += 1
            return True
        else:
            return False

    # Assign state to the function.
    fn.num_included = 0

    return fn

def filter_on_pat_ids(pat_ids: Union[str, Sequence[str]]) -> Callable[[str], bool]:
    """
    returns: a function to filter patients based on a 'pat_ids' string or list/tuple.
    args:
        pat_ids: the passed 'pat_ids' kwarg.
    """
    def fn(id):
        if ((isinstance(pat_ids, str) and (pat_ids == 'all' or id == pat_ids)) or
            ((isinstance(pat_ids, list) or isinstance(pat_ids, np.ndarray) or isinstance(pat_ids, tuple)) and id in pat_ids)):
            return True
        else:
            return False

    return fn
