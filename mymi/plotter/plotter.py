from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

from mymi import cache
from mymi import dataset

class Plotter:
    @classmethod
    def plot_patient(cls, pat_id, slice_idx, aspect=None, axes=True, figsize=(8, 8), labels=True, plane='axial', regions='all', transforms=[]):
        """
        effect: plots a CT slice with contours.
        args:
            pat_id: the patient ID.
            slice_idx: the slice to plot.
        kwargs:
            aspect: use a hard-coded aspect ratio, useful for viewing transforms.
            axes: display the pixel values on the axes.
            figsize: the size of the plot in inches.
            plane: the viewing plane.
            regions: the regions-of-interest to plot.
            transforms: the transforms to apply before plotting.
        viewplanes:
            axial: viewed from the cranium (slice_idx=0) to the caudal region.
            coronal: viewed from the anterior (slice_idx=0) to the posterior.
            sagittal: viewed from the 
        """
        # Create deterministic transforms so they're consistently applied to data
        # and labels. 
        det_transforms = [t.deterministic() for t in transforms]

        # Load CT data.
        ct_data = dataset.patient_data(pat_id, transforms=det_transforms)

        # Load labels.
        if regions is not None:
            label_data = dataset.patient_labels(pat_id, regions=regions, transforms=det_transforms)

        # Find slice in correct plane, x=sagittal, y=coronal, z=axial.
        assert plane in ('axial', 'coronal', 'sagittal')
        data_index = [
            slice_idx if plane == 'sagittal' else slice(ct_data.shape[0]),
            slice_idx if plane == 'coronal' else slice(ct_data.shape[1]),
            slice_idx if plane == 'axial' else slice(ct_data.shape[2]),
        ]
        ct_slice_data = ct_data[data_index]

        # Convert from our co-ordinate system (frontal, sagittal, longitudinal) to 
        # that required by 'imshow'.
        ct_slice_data = cls.to_image_coords(ct_slice_data, plane)

        if aspect is None:
            # Only apply aspect ratio if no transforms are being presented otherwise
            # we'll end up with skewed images.
            if len(transforms) == 0:
                summary = dataset.patient_summary(pat_id).iloc[0].to_dict()
                if plane == 'axial':
                    aspect = summary['spacing-y'] / summary['spacing-x']
                elif plane == 'coronal':
                    aspect = summary['spacing-z'] / summary['spacing-x']
                elif plane == 'sagittal':
                    aspect = summary['spacing-z'] / summary['spacing-y']
            else:
                aspect = 1

        # Plot CT data.
        plt.figure(figsize=figsize)
        plt.imshow(ct_slice_data, cmap='gray', aspect=aspect)

        if regions is not None:
            # Plot labels.
            if len(label_data) != 0:
                # Define color pallete.
                palette = plt.cm.tab20

                # Plot each label.
                show_legend = False
                for i, (lname, ldata) in enumerate(label_data):
                    # Convert data to 'imshow' co-ordinate system.
                    ldata = ldata[data_index]
                    ldata = cls.to_image_coords(ldata, plane)

                    # Skip label if not present on this slice.
                    if ldata.max() == 0:
                        continue
                    show_legend = True
                    
                    # Create binary colormap for each label.
                    colours = [(1.0, 1.0, 1.0, 0), palette(i)]
                    label_cmap = ListedColormap(colours)

                    # Plot label.
                    plt.imshow(ldata, cmap=label_cmap, aspect=aspect, alpha=0.5)
                    plt.plot(0, 0, c=palette(i), label=lname)

                # Turn on legend.
                if show_legend: 
                    legend = plt.legend(loc=(1.05, 0.8))
                    for l in legend.get_lines():
                        l.set_linewidth(8)

        # Show axis markers.
        axes = 'on' if axes else 'off'
        plt.axis(axes)

        # Determine number of slices.
        if plane == 'axial':
            num_slices = ct_data.shape[2]
        elif plane == 'coronal':
            num_slices = ct_data.shape[1]
        elif plane == 'sagittal':
            num_slices = ct_data.shape[0]

        # Add title.
        plt.title(f"{plane.capitalize()} slice: {slice_idx}/{num_slices - 1}")

        plt.show()

    @classmethod
    def to_image_coords(cls, data, plane):
        if plane == 'axial':
            data = np.transpose(data)
        elif plane in ('coronal', 'sagittal'):
            data = np.rot90(data)

        return data

    @classmethod
    def plot_batch(cls, input, slice_idx, label=None, pred=None, aspect=1.0, figsize=(8, 8), full_label=False, num_images='all', plane='axial'):
        """
        effect: plots a training batch.
        args:
            input: the input data.
            slice_idx: the slice to show.
        kwargs:
            label: the label data.
            num_images: number of images from the batch to plot.
            pred: the predicted segmentation mask.
            figsize: the plot figure size.
            full_label: should we should full label mask or perimeter?
            plane: the viewing plane.
        """
        # Get input data.
        assert plane in ('axial', 'coronal', 'sagittal')
        num_images = len(input) if num_images == 'all' else num_images
        input_data = input[:num_images]

        # Add prediction.

        # Add label.
        if label is not None:
            # Get subset of images.
            label_data = label[:num_images]

        # Plot data.
        _, axs = plt.subplots(num_images, figsize=figsize)
        if num_images == 1: axs = [axs]

        for i in range(num_images):
            # Plot input slice data.
            if plane == 'axial':
                slice_input_data = np.transpose(input_data[i, :, :, slice_idx])
            elif plane == 'coronal':
                slice_input_data = np.rot90(input_data[i, :, slice_idx, :])
            elif plane == 'sagittal':
                slice_input_data = np.rot90(input_data[i, slice_idx, :, :])

            axs[i].imshow(slice_input_data, aspect=aspect, cmap='gray')

            # Get pred slice data.

            # Plot label slice data.
            if label is not None:
                if plane == 'axial':
                    slice_label_data = np.transpose(label_data[i, :, :, slice_idx])
                elif plane == 'coronal':
                    slice_label_data = np.rot90(label_data[i, :, slice_idx, :])
                elif plane == 'sagittal':
                    slice_label_data = np.rot90(label_data[i, slice_idx, :, :])

                # Get binary perimeter.
                if full_label:
                    colour = (0.12, 0.47, 0.7, 1.0)
                else:
                    colour = (1.0, 1.0, 1.0e-5, 1.0)
                    slice_label_data = cls.binary_perimeter(slice_label_data)

                # Create binary colormap.
                colours = [(1.0, 1.0, 1.0, 0), colour]
                label_cmap = ListedColormap(colours)

                axs[i].imshow(slice_label_data, aspect=aspect, cmap=label_cmap)
        
        plt.show()

    @classmethod
    def binary_perimeter(cls, mask):
        """
        returns: the edge pixels of the mask.
        args:
            mask: a 2D image mask.
        """
        mask_perimeter = torch.zeros_like(mask, dtype=bool)
        x_dim, y_dim = mask.shape
        for i in range(x_dim):
            for j in range(y_dim):
                # Check if edge pixel.
                if (mask[i, j] == 1 and 
                    ((i == 0 or i == x_dim - 1) or
                    (j == 0 or j == y_dim - 1) or
                    i != 0 and mask[i - 1, j] == 0 or 
                    i != x_dim - 1 and mask[i + 1, j] == 0 or
                    j != 0 and mask[i, j - 1] == 0 or
                    j != y_dim - 1 and mask[i, j + 1] == 0)):
                    mask_perimeter[i, j] = 1

        return mask_perimeter