from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from attrs import frozen, field
from attrs.validators import instance_of
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from scipy.special import jn_zeros

from .._histogram import Histogram
from .._simulation import Simulation
from ...ensembles import ManyBodyEnsemble
