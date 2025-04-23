# rmtpy.simulations.evolve_cdo.py
"""
This module contains programs for performing Monte Carlo simulations to obtain the time evolutiovn of chaotic density operators (CDOs).
It is grouped into the following sections:
    1. Imports
    2. Plotting Functions
    3. Evolve CDO Class
    4. Main Function
"""


# =============================
# 1. Imports
# =============================
# Standard library imports
import os
import re
from argparse import ArgumentParser
from ast import literal_eval
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent
from time import time

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullLocator
from psutil import virtual_memory
