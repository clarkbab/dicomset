import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject

from mymi.dataset.processed import ProcessedPartition
from mymi.transforms import crop_or_pad_3D
from mymi import types

class PatchLoader:
    @staticmethod
    def build(
        partition: ProcessedPartition,
        patch_size: types.ImageSize3D,
        region: str,
        batch_size: int = 1,
        num_workers: int = 1,
        p_region: float = 1,
        shuffle: bool = True,
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None) -> torch.utils.data.DataLoader:
        """
        returns: a data loader.
        args:
            partition: the dataset partition.
            patch_size: the patch size.
            region: the single region to load patches for.
        kwargs:
            batch_size: the number of images in the batch.
            num_workers: the number of CPUs for data loading.
            p_region: proportion of patches containing the region.
            shuffle: shuffle the data.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        ds = LoaderDataset(partition, patch_size, p_region=p_region, region=region, spacing=spacing, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=ds, num_workers=num_workers, shuffle=shuffle)

class LoaderDataset(Dataset):
    def __init__(
        self,
        partition: ProcessedPartition,
        patch_size: types.ImageSize3D,
        region: str,
        p_region: float = 1,
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None):
        """
        args:
            partition: the dataset partition.
            patch_size: the patch size.
            region: load patients with this region.
        kwargs:
            p_region: proportion of patches containing the region.
            spacing: the voxel spacing.
            transform: transformations to apply.
        """
        self._p_region = p_region
        self._partition = partition
        self._patch_size = patch_size
        self._region = region
        self._spacing = spacing
        self._transform = transform
        if transform:
            assert spacing is not None, 'Spacing is required when transform applied to dataloader.'

        # Filter samples by requested regions.
        samples = partition.list_samples()
        samples = list(filter(lambda i: partition.sample(i).has_one_region(region), samples))

        # Record number of samples.
        self._num_samples = len(samples)

        # Map loader indices to dataset indices.
        self._index_map = dict(zip(range(self._num_samples), samples))

    def __len__(self):
        """
        returns: number of samples in the partition.
        """
        return self._num_samples

    def __getitem__(
        self,
        index: int):
        """
        returns: an (input, label) pair from the dataset.
        args:
            index: the item to return.
        """
        # Load data.
        input, label = self._partition.sample(self._index_map[index]).pair(regions=self._region)
        label = label[self._region]

        # Perform transform.
        if self._transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self._spacing[0], 0, 0, 0],
                [0, self._spacing[1], 0, 0],
                [0, 0, self._spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject = Subject(input=input, label=label)

            # Transform the subject.
            output = self._transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

        # Roll the dice.
        if np.random.binomial(1, self._p_region):
            input, label = self._get_region_patch(input, label)
        else:
            input, label = self._get_background_patch(input, label)

        # Add 'channel' dimension.
        input = np.expand_dims(input, axis=0)

        # Convert to half precision.
        input = input.astype(np.half)

        return input, label

    def _get_region_patch(
        self,
        input: np.ndarray,
        label: np.ndarray) -> np.ndarray:
        """
        returns: a patch around the OAR.
        args:
            input: the input data.
            label: the label data.
        """
        # Find foreground voxels.
        fg_voxels = np.argwhere(label != 0)
        
        # Choose randomly from the foreground voxels.
        fg_voxel_idx = np.random.choice(len(fg_voxels))
        centre_voxel = fg_voxels[fg_voxel_idx]

        # Determine min/max indices of the patch.
        shape_diff = np.array(self._patch_size) - 1
        lower_add = np.ceil(shape_diff / 2).astype(int)
        mins = centre_voxel - lower_add
        maxs = mins + self._patch_size

        # Crop or pad the volume.
        input = crop_or_pad_3D(input, (mins, maxs), fill=input.min())
        label = crop_or_pad_3D(label, (mins, maxs))

        return input, label

    def _get_background_patch(
        self,
        input: np.ndarray,
        label: np.ndarray) -> np.ndarray:
        """
        returns: a random patch from the volume.
        args:
            input: the input data.
            label: the label data.
        """
        # Choose a random voxel.
        centre_voxel = tuple(map(np.random.randint, self._patch_size))

        # Determine min/max indices of the patch.
        shape_diff = np.array(self._patch_size) - 1
        lower_add = np.ceil(shape_diff / 2).astype(int)
        mins = centre_voxel - lower_add
        maxs = mins + self._patch_size

        # Crop or pad the volume.
        input = crop_or_pad_3D(input, (mins, maxs), fill=input.min())
        label = crop_or_pad_3D(label, (mins, maxs))

        return input, label