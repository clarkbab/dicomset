import logging
import numpy as np
import os
import pandas as pd
from skimage.draw import polygon
import sys
from torchio import LabelMap, ScalarImage, Subject
from tqdm import tqdm
from typing import Optional

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import config
from mymi.transforms import resample_3D
from mymi import types

from ...processed import create as create_processed_dataset
from ...processed import destroy as destroy_processed_dataset
from ...processed import list as list_processed_datasets
from .nifti_dataset import NIFTIDataset

def process_nifti(
    dataset: str,
    dest_dataset: Optional[str] = None,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_validation: float = 0.2,
    random_seed: int = 42,
    spacing: Optional[types.ImageSpacing3D] = None):
    """
    effect: processes a NIFTI dataset and partitions it into train/validation/test
        partitions for training.
    args:
        the dataset to process.
    kwargs:
        p_test: the proportion of test objects.
        p_train: the proportion of train objects.
        p_validation: the proportion of validation objects.
        regions: the regions to process.
        spacing: resample to the desired spacing.
    """
    # Load objects.
    ds = NIFTIDataset(dataset)
    ids = list(ds.region_names()['id'].unique())
    logging.info(f"Found {len(ids)} objects.")

    # Shuffle objects.
    np.random.seed(random_seed) 
    np.random.shuffle(ids)

    # Partition patients - rounding assigns more patients to the test set,
    # unless p_test=0, when these are assigned to the validation set.
    num_train = int(np.floor(p_train * len(ids)))
    if p_test == 0:
        num_validation = len(ids) - num_train
    else:
        num_validation = int(np.floor(p_validation * len(ids)))
    num_train = int(np.floor(p_train * len(ids)))
    num_validation = int(np.floor(p_validation * len(ids)))
    train_ids = ids[:num_train]
    validation_ids = ids[num_train:(num_train + num_validation)]
    test_ids = ids[(num_train + num_validation):]
    logging.info(f"Num objects per partition: {len(train_ids)}/{len(validation_ids)}/{len(test_ids)} for train/validation/test.")

    # Destroy old dataset if present.
    name = dest_dataset if dest_dataset else dataset
    if name in list_processed_datasets():
        destroy_processed_dataset(name)

    # Create dataset.
    proc_ds = create_processed_dataset(name)

    # Write data to each partition.
    partitions = ['train', 'validation', 'test']
    partition_ids = [train_ids, validation_ids, test_ids]
    for partition, ids in zip(partitions, partition_ids):
        logging.info(f"Writing '{partition}' objects..")

        # TODO: implement normalisation.

        # Write each object to partition.
        for id in tqdm(ids):
            # Get available requested regions.
            obj_regions = ds.object(id).region_names()

            # Load data.
            input = ds.object(id).ct_data()
            labels = ds.object(id).region_data(regions=obj_regions)

            # Resample data if requested.
            if spacing is not None:
                old_spacing = ds.object(id).ct_spacing()
                input = resample_3D(input, old_spacing, spacing)

            # Save input data.
            index = proc_ds.partition(partition).create_input(id, input)

            # Save label data.
            for region, label in labels.items():
                proc_ds.partition(partition).create_label(index, region, label)
