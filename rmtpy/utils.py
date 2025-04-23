# rmtpy.utils.py
"""
This module contains functions for plotting and other utilities used in the RMTPy package.
It is grouped into the following sections:
    1. Imports
    2. Ensemble Registry
    3. Utility Functions
    4. Plotting Functions
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
import re
from ast import literal_eval
from importlib import import_module
from pathlib import Path
from typing import Any, Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# =============================
# 2. Ensemble Registry
# =============================
# Instantiate ensemble registry
_ENSEMBLE_REGISTRY = {}

# Loop through all modules in directory
for file in Path(__file__).parent.glob("ensembles/*.py"):
    # Skip files beginning with an underscore
    if file.name.startswith("_"):
        continue

    # Import module
    module = import_module(f"rmtpy.ensembles.{file.stem}")

    # Get ensemble class
    _ENSEMBLE = getattr(module, module.class_name)

    # Register ensemble class
    _ENSEMBLE_REGISTRY[file.stem] = _ENSEMBLE


# =============================
# 3. Utility Functions
# =============================
def get_ensemble(name: str, **kwargs) -> object:
    """
    Get an ensemble from the registry.

    Parameters
    ----------
    name : str
        Name of the ensemble.
    **kwargs : dict
        Additional arguments to pass to the ensemble constructor.

    Returns
    -------
    object
        The ensemble object.
    """
    if name not in _ENSEMBLE_REGISTRY:
        raise ValueError(f"Ensemble '{name}' not found in registry.")

    return _ENSEMBLE_REGISTRY[name](**kwargs)


def _ensemble_from_path(path: str, file_name: str) -> object:
    """
    Determines the ensemble from the given path of data file.

    Parameters
    ----------
    path : str
        Path to the data file.
    file_name : str
        Name of the data file.

    Returns
    -------
    object
        Initialized ensemble object.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Check if path is named correctly
    if os.path.basename(path) != file_name:
        raise ValueError(f"File name must be '{file_name}'")

    # Initialize metadata dictionary
    metadata = {}

    # Grabs ensemble name from path
    try:
        ensemble_name = next(datum for datum in path.split("/")[2:])
    except StopIteration:
        raise ValueError(f"Empty path given.")

    # Extract ensemble name and metadata from path
    for part in Path(path).parts:
        datum = re.fullmatch(r"(?P<key>\w+)=(?P<val>.+)", part)
        if datum is not None:
            metadata[datum.group("key")] = literal_eval(datum.group("val"))

    # Pop realizations from metadata
    metadata.pop("realizs", None)

    # Return initialized ensemble
    return get_ensemble(ensemble_name, **metadata)


# =============================
# 4. Plotting Functions
# =============================
def _initialize_plot(dataclass: Any, data_path: str) -> Tuple[
    Any,
    dict,
    bool,
    Figure,
    Axes,
]:
    """
    Initializes the plot by loading data and creating a figure and axis.

    Parameters
    ----------
    dataclass : Any
        Configuration class containing plot parameters.
    data_path : str
        Path to the data file containing histogram data.

    Returns
    -------
    Tuple[Any, dict, bool, Figure, Axes]
        Tuple containing the ensemble object, data dictionary, unfold flag, figure, and axis.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the file name does not match the expected name or if the ensemble name is not found in the path.
    """
    # Check if data is unfolded
    unfold = "unfolded" in os.path.basename(data_path)

    # Reads results path and extracts ensemble
    if not unfold:
        ensemble = _ensemble_from_path(data_path, dataclass.data_filename)
    else:
        ensemble = _ensemble_from_path(data_path, dataclass.unfolded_data_filename)

    # Load data from file
    data = np.load(data_path)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Return ensemble, data, fig, ax
    return ensemble, data, unfold, fig, ax


def _create_plot(
    dataclass: Any,
    data_path: str,
    legend_title: str,
    fig: Figure,
    ax: Axes,
    unfold: bool,
) -> None:
    """
    Creates a plot with specified parameters and saves it to a file.

    Parameters
    ----------
    dataclass : Any
        Configuration class containing plot parameters.
    data_path : str
        Path to the data file containing histogram data.
    legend_title : str
        Title for the legend.
    fig : Figure
        Figure object for the plot.
    ax : Axes
        Axes object for the plot.
    unfold : bool
        Flag indicating whether the data is unfolded or not.
    """
    # Set x-axis labels
    ax.set_xlabel(dataclass.unfolded_xlabel if unfold else dataclass.xlabel)

    # Set x-tick labels
    if not unfold and dataclass.has_xticklabels:
        # Use custom tick labels
        ax.set_xticklabels(
            dataclass.xticklabels,
            fontsize=dataclass.ticklabel_fontsize,
        )
    elif unfold and dataclass.has_unfolded_xticklabels:
        # Use custom unfolded tick labels
        ax.set_xticklabels(
            dataclass.unfolded_xticklabels,
            fontsize=dataclass.ticklabel_fontsize,
        )
    else:
        # Resize default x-tick labels
        ax.tick_params(axis="x", labelsize=dataclass.ticklabel_fontsize)

    # Set y-axis labels
    ax.set_ylabel(dataclass.unfolded_ylabel if unfold else dataclass.ylabel)

    # Set y-tick labels
    if not unfold and dataclass.has_yticklabels:
        # Use custom tick labels
        ax.set_yticklabels(dataclass.yticklabels, fontsize=dataclass.ticklabel_fontsize)
    elif unfold and dataclass.has_unfolded_yticklabels:
        # Use custom unfolded tick labels
        ax.set_yticklabels(
            dataclass.unfolded_yticklabels,
            fontsize=dataclass.ticklabel_fontsize,
        )
    else:
        # Resize default y-tick labels
        ax.tick_params(axis="y", labelsize=dataclass.ticklabel_fontsize)

    # Set tick marks all around and inward
    ax.tick_params(
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        which="both",
        length=dataclass.tick_length,
    )

    # Set spine widths
    for spine in ax.spines.values():
        spine.set_linewidth(dataclass.axes_width)

    # Set legend handles and labels based on unfolding
    if not unfold:
        legend_handles = dataclass.legend_handles
        legend_labels = dataclass.legend_labels
    else:
        legend_handles = dataclass.unfolded_legend_handles
        legend_labels = dataclass.unfolded_legend_labels

    # Create legend
    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title=legend_title,
        loc=dataclass.legend_location,
        bbox_to_anchor=dataclass.legend_bbox,
        fontsize=dataclass.legend_fontsize,
        title_fontsize=dataclass.legend_title_fontsize,
        frameon=dataclass.legend_frameon,
    )
    legend._legend_box.align = dataclass.legend_textalignment  # type: ignore[attr-defined]

    # Store plot file name
    plot_file = dataclass.unfolded_plot_filename if unfold else dataclass.plot_filename

    # Create plot path from data path
    data_Path = Path(data_path)
    plot_dir = data_Path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / plot_file

    # Save plot to file
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    # Close plot
    plt.close(fig)
