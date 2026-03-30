from mymi.regions import regions_to_list
import numpy as np
import os
from typing import List, Literal, Tuple

from ....dicom import Dataset, DicomDataset, DicomRtStructSeries, SeriesID, config, dicom, list, pd, property, regions
from ....regions_map import RegionsMap
from .image import NiftiImageSeries

class NiftiRegionsSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset: 'NiftiDataset',
        pat: 'NiftiPatient',
        study: 'NiftiStudy',
        id: SeriesID,
        index: pd.DataFrame | None = None,
        regions_map: RegionsMap | None = None,
        ) -> None:
        super().__init__('regions', dataset, pat, study, id, index=index)
        extensions = ['.nii', '.nii.gz', '.nrrd']
        dirpath = os.path.join(config.directories.datasets, 'nifti', self._dataset.id, 'data', 'patients', self._pat.id, self._study.id, self._modality, self._id)
        if not os.path.exists(dirpath):
            raise ValueError(f"No regions series '{self._id}' found for study '{self._study.id}'. Dirpath: {dirpath}")
        self.__dirpath = dirpath
        self.__regions_map = regions_map

    @alias_kwargs([
        ('r', 'regions'),
    ])
    def data(
        self,
        regions: Region | List[Region] | Literal['all'] = 'all',
        regions_ignore_missing: bool = True,
        return_regions: bool = False,
        **kwargs,
        ) -> LabelVolumeBatch | Tuple[LabelVolumeBatch, List[Region]]:
        regions = regions_to_list(regions, literals={ 'all': self.list_regions })

        # Get region names.
        regions_filtered = []
        for r in regions:
            if not self.has_region(r):
                if regions_ignore_missing:
                    continue
                else:
                    raise ValueError(f'Region {r} not found in image {self.id}.')
            regions_filtered.append(r)

        # Add regions data.
        regions_data = None    # We don't know the shape yet.
        for i, r in enumerate(regions_filtered):
            # Load region from disk.
            # If multiple regions have been mapped to the same ID, then get the union of these regions.
            filepaths = self.filepaths(r)
            ds = []
            for f in filepaths:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    d, _ = load_nifti(f)
                elif f.endswith('.nrrd'):
                    d, _ = load_nrrd(f)
                else:
                    raise ValueError(f'Unsupported file format: {f}')
                ds.append(d)
            if regions_data is None:
                regions_data = np.zeros((len(regions_filtered), *d.shape), dtype=bool)
            regions_data[i] = np.sum(ds, axis=0).clip(0, 1).astype(bool)

        if return_regions:
            return regions_data, regions_filtered
        else:
            return regions_data

    @property
    def dicom(self) -> DicomRtStructSeries:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset.id) & (index['patient-id'] == self._pat.id) & (index['study-id'] == self._study.id) & (index['series-id'] == self._id) & (index['modality'] == 'regions')].drop_duplicates()
        assert len(index) == 1, f"Expected exactly one matching row in index for series {self.id}, but found {len(index)}. Index: {index}"
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).rtstruct_series(row['dicom-series-id'])

    def filepaths(
        self,
        regions: Region | List[Region],
        regions_ignore_missing: bool = True,
        ) -> List[FilePath]:
        regions = arg_to_list(regions, str)
        if not regions_ignore_missing and not self.has_region(regions):
            raise ValueError(f'Regions {regions} not found in series {self.id}.')
        regions = [r for r in regions if self.has_region(r)]  # Filter out missing regions.
        # Region mapping is many-to-one, so we could get multiple files on disk for the same mapped region.
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        disk_ids = self.__regions_map.inv_map_region(regions, disk_regions=self.list_regions(use_mapping=False)) if self.__regions_map is not None else regions
        disk_ids = arg_to_list(disk_ids, str)
        # Check all possible file extensions.
        filepaths = [os.path.join(self.__dirpath, f'{i}{e}') for i in disk_ids for e in image_extensions if os.path.exists(os.path.join(self.__dirpath, f'{i}{e}'))]
        return filepaths

    def has_region(
        self,
        regions: RegionIDs,
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_regions(**kwargs)
        regions = arg_to_list(regions, RegionID)
        n_overlap = len(np.intersect1d(regions, all_ids))
        return n_overlap > 0 if any else n_overlap == len(regions)

    @alias_kwargs(('um', 'use_mapping'))
    def list_regions(
        self,
        region: RegionIDs = 'all',
        use_mapping: bool = True) -> List[Region]:
        # Load regions from filenames.
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        ids = os.listdir(self.__dirpath)
        ids = [i.replace(e, '') for i in ids for e in image_extensions if i.endswith(e)]

        # Apply region mapping.
        if use_mapping and self.__regions_map is not None:
            ids = [self.__regions_map.map_region(i) if self.__regions_map is not None else i for i in ids]

        # Filter on 'only'.
        if region != 'all':
            regions = regions_to_list(region)
            ids = [r for r in ids if r in regions]

        # Sort regions.
        ids = list(sorted(ids))

        return ids

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
