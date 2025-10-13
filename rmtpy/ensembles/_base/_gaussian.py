# rmtpy/ensembles/_gaussian.py

# Postponed evaluation of annotations
from __future__ import annotations

# Third-party imports
import numpy as np
from attrs import field, frozen

# Local application imports
from ._manybody import ManyBodyEnsemble


# ----------------------------
# Gaussian Ensemble Base Class
# ----------------------------
@frozen(kw_only=True)
class GaussianEnsemble(ManyBodyEnsemble):

    # Complex standard deviation of matrix entries
    sigma: float = field(init=False, repr=False)

    # Derive sigma from dimension and ground state energy
    @sigma.default
    def __sigma_default(self) -> float:
        """Calculate the complex standard deviation of matrix entries."""

        return self.E0 / np.sqrt(2 * self.dim)

    def pdf(self, eigval: np.ndarray) -> np.ndarray:
        """Wigner semicircle probability density function."""

        # Normalize eigenvalues
        x = eigval / self.E0

        # Initialize PDF array
        pdf = np.zeros_like(x, dtype=self.real_dtype)

        # Create mask for eigenvalues
        mask = np.abs(x) < 1.0

        # Calculate PDF for eigenvalue array
        pdf[mask] = np.sqrt(1 - x[mask] * x[mask])
        pdf[mask] *= 2 / np.pi / self.E0

        # Return PDF array
        return pdf

    def cdf(self, eigval: np.ndarray) -> np.ndarray:
        """Cumulative distribution function of Wigner semicircle PDF."""

        # Normalize eigenvalues and clip to range [-1, 1]
        x = np.clip(eigval / self.E0, -1.0, 1.0)

        # Build CDF array
        cdf = np.sqrt(1 - x * x)
        cdf *= x
        cdf += np.arcsin(x)
        cdf /= np.pi
        cdf += 0.5

        # Return CDF array
        return cdf
