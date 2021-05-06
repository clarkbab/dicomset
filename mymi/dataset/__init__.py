import logging
import os
import sys

from .dicom import DicomDataset

MYMI_DATA = os.environ['MYMI_DATA']
DATASETS_PATH = os.path.join(MYMI_DATA, 'datasets')
DEFAULT_ACTIVE = 'HEAD-NECK-RADIOMICS-HN1'

# Create dataset.
active = DicomDataset(DEFAULT_ACTIVE)

def select(
    name: str):
    """
    effect: sets the new dataset as active.
    args:
        name: the name of the new dataset.
    """
    # Check if the dataset exists.
    dataset_path = os.path.join(DATASETS_PATH, name)
    if os.path.exists(dataset_path):
        global active
        active = DicomDataset(name)
    else:
        raise ValueError(f"Dataset '{name}' not found.")

def ct_summary(*args, **kwargs):
    return active.ct_summary(*args, **kwargs)

def list_patient(*args, **kwargs):
    return active.list_patients(*args, **kwargs)
    
def patient(*args, **kwargs):
    return active.patient(*args, **kwargs)

def ct_summaries(*args, **kwargs):
    return active.ct_summaries(*args, **kwargs)

def ct_statistics(*args, **kwargs):
    return active.ct_statistics(*args, **kwargs)

def get_rtstruct(*args, **kwargs):
    return active.get_rtstruct(*args, **kwargs)

def labels(*args, **kwargs):
    return active.labels(*args, **kwargs)

def list_ct(*args, **kwargs):
    return active.list_ct(*args, **kwargs)

def list_patients():
    return active.list_patients()

def label_count(*args, **kwargs):
    return active.label_count(*args, **kwargs)

##
# Processed dataset API.
##

def class_frequencies(*args, **kwargs):
    return active.class_frequencies(*args, **kwargs)

def input(*args, **kwargs):
    return active.input(*args, **kwargs)

def label(*args, **kwargs):
    return active.label(*args, **kwargs)

def list_samples(*args, **kwargs):
    return active.list_samples(*args, **kwargs)

def sample(*args, **kwargs):
    return active.sample(*args, **kwargs)
