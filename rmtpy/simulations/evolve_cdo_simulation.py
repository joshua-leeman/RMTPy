# rmtpy/simulations/evolve_cdo_simulation.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
from attrs import frozen, field
from scipy.linalg import eigvalsh, eigh
from scipy.special import jn_zeros

# Local application imports
from ._simulation import Simulation
from ..ensembles import ManyBodyEnsemble


# ---------------------------------
# CDO Evolution Simulation Function
# ---------------------------------


# ------------------------------
# CDO Evolution Simulation Class
# ------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class CDOEvolution(Simulation):

    # Unfold flag

    # Initial state

    # Evolved states data

    def realize_monte_carlo(self) -> None:
        pass

        # Loop over diagonalization realizations and store evolved pure states

        # Chunk evolved states data and save pieces to disk

    # Calculate:
    # Probabilities
    # Observable expectation
    # Classical purity
    # Quantum purity
    # von Neumann entropy
    # Single realization Kullback-Leibler divergence

    # Single run function to combine all calculations
