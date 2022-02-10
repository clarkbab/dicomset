from distutils.dir_util import copy_tree
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
from time import time
from tqdm import tqdm
from scipy.ndimage import binary_dilation
import sys

from mymi.dataset import DatasetType
from mymi.dataset.nifti import recreate as recreate_nifti
from mymi import logging
from mymi import types

def convert_to_nifti(
    dataset: 'Dataset',
    regions: types.PatientRegions = 'all',
    anonymise: bool = False) -> None:
    # Create NIFTI dataset.
    nifti_ds = recreate_nifti(dataset.name)

    logging.info(f"Converting dataset '{dataset}' to dataset '{nifti_ds}', with regions '{regions}' and anonymise '{anonymise}'.")

    # Load all patients.
    pats = dataset.list_patients(regions=regions)

    if anonymise:
        # Create CT map. Index of map will be the anonymous ID.
        map_df = pd.DataFrame(pats, columns=['patient-id'])

        # Save map.
        filename = 'map.csv'
        filepath = os.path.join(dataset.path, f'anon-nifti-map.csv')
        map_df.to_csv(filepath)

    for pat in tqdm(pats):
        # Get anonymous ID.
        if anonymise:
            anon_id = map_df[map_df['patient-id'] == pat].index.values[0]
            filename = f'{anon_id}.nii.gz'
        else:
            filename = f'{pat}.nii.gz'

        # Create CT NIFTI.
        patient = dataset.patient(pat)
        data = patient.ct_data()
        spacing = patient.ct_spacing()
        offset = patient.ct_offset()
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filepath = os.path.join(nifti_ds.path, 'data', 'ct', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Create region NIFTIs.
        pat_regions = patient.list_regions(whitelist=regions)
        region_data = patient.region_data(regions=pat_regions)
        for region, data in region_data.items():
            img = Nifti1Image(data.astype(np.int32), affine)
            filepath = os.path.join(nifti_ds.path, 'data', region, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

    # Indicate success.
    _write_flag(nifti_ds, '__CONVERT_FROM_NIFTI_END__')