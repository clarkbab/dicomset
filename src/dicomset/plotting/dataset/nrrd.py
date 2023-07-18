import numpy as np
import re
from typing import Dict, Literal, Optional, Union

from dicomset.dataset import NRRDDataset
from dicomset.types import Crop2D, PatientRegions

from ..plotter import plot_region as plot_region_base

def plot_region(
    dataset: str,
    pat_id: str,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, Crop2D]] = None,
    labels: Literal['included', 'excluded', 'all'] = 'all',
    region: Optional[PatientRegions] = None,
    region_label: Optional[Dict[str, str]] = None,     # Gives 'regions' different names to those used for loading the data.
    show_dose: bool = False,
    **kwargs) -> None:

    # Load data.
    set = NRRDDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data
    region_data = pat.region_data(labels=labels, region=region) if region is not None else None
    spacing = pat.ct_spacing
    dose_data = pat.dose_data if show_dose else None

    if centre_of is not None:
        if type(centre_of) == str:
            if region_data is None or centre_of not in region_data:
                centre_of = pat.region_data(region=centre_of)[centre_of]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = pat.region_data(region=crop)[crop]

    if region_label is not None:
        # Rename regions.
        for old, new in region_label.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre_of' and 'crop' keys.
        if type(centre_of) == str and centre_of in region_label:
            centre_of = region_label[centre_of] 
        if type(crop) == str and crop in region_label:
            crop = region_label[crop]

    # Plot.
    plot_region_base(pat_id, ct_data.shape, spacing, centre_of=centre_of, crop=crop, ct_data=ct_data, dose_data=dose_data, region_data=region_data, **kwargs)
