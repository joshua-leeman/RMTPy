# rmtpy.simulations.width_statistics.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import os
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass, field

# Third-party imports
import numpy as np
from scipy.special import jn_zeros
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt

# Local application imports
from rmtpy.simulations._mc import MonteCarlo, _parse_mc_args


# =======================================
# 2. Simulation Class
# =======================================
@dataclass(repr=False, eq=False, frozen=True, kw_only=True, slots=True)
class WidthStatistics(MonteCarlo):
    def run(self) -> None:
        # Relabel simulation attributes for clarity
        realizs = self.realizs
        dim = self.ensemble.dim
        real_dtype = self.ensemble.real_dtype

        # Initialize memory to store real parts of eigenvalues
        resons = np.empty(realizs * dim, dtype=real_dtype)

        # Initialize memory to store imaginary parts of eigenvalues
        widths = np.empty(realizs * dim, dtype=real_dtype)

        # Initialize memory to store real parts of unfolded eigenvalues
        unf_resons = np.empty(realizs * dim, dtype=real_dtype)

        # Initialize memory to store imaginary parts of unfolded eigenvalues
        unf_widths = np.empty(realizs * dim, dtype=real_dtype)

        # Loop over spectrum realizations and store complex eigenvalues
        for r, eigvals in enumerate(self.ensemble.eff_H_eigvals_stream(realizs)):
            # Store real and imaginary parts of eigenvalues
            resons[r * dim : (r + 1) * dim] = eigvals.real
            widths[r * dim : (r + 1) * dim] = -eigvals.imag

            print(-eigvals[eigvals.imag < -20])
            print(-eigvals.imag[eigvals.imag < -20])

            # Unfold resonance energies
            unf_resons_r = self.ensemble.unfold(eigvals.real)

            # Unfold widths with each centered at its corresponding resonance\
            unf_widths_r = self.ensemble.unfold(eigvals.real - eigvals.imag / 2)
            unf_widths_r -= self.ensemble.unfold(eigvals.real + eigvals.imag / 2)

            # Store unfolded real and imaginary parts of unfolded eigenvalues
            unf_resons[r * dim : (r + 1) * dim] = unf_resons_r
            unf_widths[r * dim : (r + 1) * dim] = unf_widths_r

        stat, bin_edges, _ = binned_statistic(resons, widths, statistic="mean", bins=40)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        plt.figure(figsize=(10, 6))
        # make y-axis logarithmic
        plt.yscale("log")

        plt.scatter(resons, widths, s=1, alpha=0.5, label="Data points")
        plt.plot(bin_centers, stat, label="Mean widths", color="red", linewidth=2)
        plt.show()
