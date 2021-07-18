import numpy as np
import pydicom as dcm
import torch
from torch import nn
from typing import *

from mymi import types

from .localisation import get_patient_localisation
from .patch_segmentation import get_patient_patch_segmentation

def get_patient_two_stage_segmentation(
    id: types.PatientID,
    localiser: nn.Module,
    localiser_size: types.ImageSize3D,
    localiser_spacing: types.ImageSpacing3D,
    segmenter: nn.Module,
    segmenter_size: types.ImageSize3D,
    segmenter_spacing: types.ImageSpacing3D,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    use_postprocessing: bool = True) -> np.ndarray:
    """
    returns: the patient segmentation.
    args:
        ds: the dataset name.
        patient: the patient ID.
        localiser: the localiser model.
        localiser_size: the input size of the localiser network.
        localiser_spacing: the voxel spacing of the localiser network input layer.
        segmenter: the segmenter model.
        segmenter_size: the input size of the segmenter network.
        segmenter_spacing: the voxel spacing of the segmenter network input layer.
    kwargs:
        clear_cache: force the cache to clear.
        device: the device to perform network calcs on.
        use_postprocessing: apply postprocessing steps.
    """
    # Get the OAR bounding box.
    bounding_box = get_patient_localisation(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device, use_postprocessing=use_postprocessing)

    # Get segmentation prediction.
    seg = get_patient_patch_segmentation(id, bounding_box, segmenter, segmenter_size, segmenter_spacing, clear_cache=clear_cache, device=device, use_postprocessing=use_postprocessing)

    return seg