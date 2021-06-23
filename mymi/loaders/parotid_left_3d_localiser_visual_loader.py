import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Sampler
from torchio import Compose, LabelMap, ScalarImage, Subject

from mymi import config

class ParotidLeft3DLocaliserVisualLoader:
    @staticmethod
    def build(batch_size=1, num_batches=5, seed=42, spacing=None, transform=None):
        """
        returns: a data loader.
        kwargs:
            batch_size: the number of images in a batch.
            num_batches: how many batches this loader should generate.
            seed: random number generator seed.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DLocaliserVisualDataset(spacing=spacing, transform=transform)

        # Create sampler.
        sampler = ParotidLeft3DLocaliserVisualSampler(dataset, num_batches * batch_size, seed)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)

class ParotidLeft3DLocaliserVisualDataset(Dataset):
    def __init__(self, spacing=None, transform=None):
        """
        returns: a dataset.
        kwargs:
            spacing: the voxel spacing.
            transforms: an array of augmentation transforms.
        """
        self.spacing = spacing
        self.transform = transform
        if transform:
            assert spacing, 'Spacing is required when transform applied to dataloader.'

        # Load paths to all samples.
        self.data_dir = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'validation')
        samples = np.reshape([os.path.join(self.data_dir, p) for p in sorted(os.listdir(self.data_dir))], (-1, 2))

        self.num_samples = len(samples)
        self.samples = samples

    def __len__(self):
        """
        returns: number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        returns: an (input, label) pair from the dataset.
        idx: the item to return.
        """
        # Get data and label paths.
        input_path, label_path = self.samples[idx]

        # Load data and label.
        f = open(input_path, 'rb')
        input = np.load(f)
        f = open(label_path, 'rb')
        label = np.load(f)

        # Perform transform.
        if self.transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self.spacing[0], 0, 0, 0],
                [0, self.spacing[1], 0, 0],
                [0, 0, self.spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject = Subject(input=input, label=label)

            # Transform the subject.
            output = self.transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

        return input, label

class ParotidLeft3DLocaliserVisualSampler(Sampler):
    def __init__(self, dataset, num_images, seed):
        self.dataset_length = len(dataset)
        self.num_images = num_images
        self.seed = seed

    def __iter__(self):
        # Set random seed for repeatability.
        np.random.seed(self.seed)

        # Get random subset of indices.
        indices = list(range(self.dataset_length))
        np.random.shuffle(indices)
        indices = indices[:self.num_images]

        return iter(indices)

    def __len__(self):
        return self.num_images
