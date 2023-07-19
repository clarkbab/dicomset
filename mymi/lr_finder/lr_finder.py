import json
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple, Union

from mymi import config
from mymi.utils import arg_to_list

def load_data(
    model_name: str,
    run_name: str) -> Dict[str, np.ndarray]:
    filepath = os.path.join(config.directories.models, model_name, run_name, 'lr-finder.json')
    return json.load(open(filepath))

def suggestion(
    model_name: str,
    run_name: str,
    skip_end: int = 0,
    skip_start: int = 0,
    smooth: bool = False,
    smooth_kernel_sd: float = 1) -> float:
    # Get loss data.
    results = load_data(model_name, run_name)
    losses = results['loss']

    # Remove 'nan' losses.
    real_idxs = np.argwhere(~np.isnan(losses)).flatten()
    lr = list(np.array(lr)[real_idxs])
    losses = list(np.array(losses)[real_idxs])

    # Smooth the signal.
    if smooth:
        losses = gaussian_filter1d(losses, smooth_kernel_sd)

    # Remove start/end points.
    if skip_end != 0:
        losses = losses[skip_start:-skip_end]
    else:
        losses = losses[skip_start:]

    # Get minimum gradient.
    min_grad_idx = np.gradient(losses).argmin()
    min_grad_idx = skip_start + min_grad_idx

    # Get associated learning rate.
    lr = results['lr'][min_grad_idx]

    return lr

def plot(
    model_name: str,
    run_name: Union[str, List[Union[str, List[str]]]],
    label: Optional[Union[str, List[str]]] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = (1, 1),
    skip_end: int = 0,
    skip_start: int = 0,
    smooth: bool = False,
    smooth_kernel_sd: float = 1,
    suggested_lr: bool = True,
    vline: Optional[Union[float, List[float]]] = None,
    y_lim: Tuple[Optional[float], Optional[float]] = (None, None)) -> None:
    run_names = arg_to_list(run_name, str)
    labels = arg_to_list(label, str)
    if labels is not None and len(labels) != len(run_names):
        raise ValueError(f"Must pass same number of labels ({len(labels)}) as run_names ({len(run_names)}).")

    # Load run data.
    losses = {}
    for i, run_name in enumerate(run_names):
        # Multiple runs can be specified to average them.
        averaged_runs = arg_to_list(run_name, str)

        # Determine run name.
        if len(averaged_runs) > 1:
            run_name = '/'.join(averaged_runs)

        # Load averaged losses.
        a_losses = []
        a_lrs = []
        for a_run in averaged_runs:
            results = load_data(model_name, a_run)
            lrs = results['lr']
            if len(a_lrs) > 0:
                if lrs != a_lrs:
                    raise ValueError(f'Averaged runs must be evaluated over same learning rates.')
            a_lrs = lrs
            losses = results['loss']
            a_losses.append(losses)
        losses = np.vstack(a_losses).mean(axis=0)

        # Remove 'nan' losses.
        non_nan_idx = np.argwhere(~np.isnan(losses)).flatten()
        lrs = list(np.array(lrs)[non_nan_idx])
        losses = list(np.array(losses)[non_nan_idx])
        
        # Smooth the signal.
        if smooth:
            losses = gaussian_filter1d(losses, smooth_kernel_sd)

        # Remove start/end points.
        if skip_end != 0:
            sugg_losses  = losses[skip_start:-skip_end] 
        else:
            sugg_losses = losses[skip_start:]

        # Load suggestion.
        min_grad_idx = np.gradient(sugg_losses).argmin()
        min_grad_idx = skip_start + min_grad_idx
        sugg_lr = lrs[min_grad_idx]
        sugg_loss = losses[min_grad_idx]

        # Plot.
        if labels is not None:
            label = labels[i]
        else:
            label = f'{run_name}: {sugg_lr:.6f}' if suggested_lr else run_name
        plt.plot(lrs, losses, label=label)
        plt.scatter(sugg_lr, sugg_loss, color='red')

    # Plot vlines.
    if vline is not None:
        vlines = arg_to_list(vline, float)
        for vline in vlines:
            plt.axvline(vline, linestyle='--', color='red')

    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.xscale('log')
    plt.title('LR Find')
    plt.legend(bbox_to_anchor=legend_bbox_to_anchor)
    plt.ylim(y_lim)
    plt.grid()
    plt.show()
