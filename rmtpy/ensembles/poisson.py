# rmtpy/ensembles/poisson.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
from collections.abc import Iterator

# Third-party imports
import numpy as np
from attrs import field, frozen
from scipy.linalg import eigh

# Local application imports
from ._base import ManyBodyEnsemble, converter


# ----------------
# Poisson Ensemble
# ----------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Poisson(ManyBodyEnsemble):

    # Standard deviation of eigenvalues
    sigma: float = field(init=False, repr=False)

    # Flag of ensemble from which to draw eigenvectors
    eigvecs_flag: str = field(default="GUE")

    # Ensemble from which to draw eigenvectors
    eigvecs_ensemble: ManyBodyEnsemble = field(init=False, repr=False)

    @sigma.default
    def __sigma_default(self) -> float:
        """Default value for sigma."""

        # Calculate standard deviation based on E0
        return 2 * self.E0

    def __attrs_post_init__(self) -> None:
        """Post-initialization method to initialize eigenvector ensemble."""

        # Unstructure Poisson class instance to dictionary
        dic = converter.unstructure(self)

        # Change name in dictionary to eigvecs_flag
        dic["name"] = self.eigvecs_flag

        # Remove eigvecs_flag from dictionary
        dic["args"].pop("eigvecs_flag")

        # Create ensemble instance from modified dictionary
        ens = converter.structure(dic, ManyBodyEnsemble)

        # Store ensemble instance in eigvecs_ensemble attribute
        object.__setattr__(self, "eigvecs_ensemble", ens)

    def generate_matrix(
        self, out: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Generate a random matrix from the Poisson ensemble."""

        # Pass since matrix generation is not implemented
        raise NotImplementedError(
            "Matrix generation is not implemented for the Poisson ensemble."
        )

    def eigsys_stream(self, realizs: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator to stream eigensystem realizations."""

        # Alias random number generator
        rng = self.rng

        # Alias data types of eigenvalues and eigenvectors
        rdtype = self.real_dtype.type
        cdtype = self.dtype.type

        # Alias dimension of matrix
        d = self.dim

        # Alias standard deviation of eigenvalues
        std = self.sigma

        # Alias ensemble for generating eigenvectors
        eigvecs_ens = self.eigvecs_ensemble

        # =================================================

        # Allocate memory for random eigenvalues
        eigvals = np.empty(d, rdtype, order="F")

        # If ensemble flag is GOE, allocate memory for real eigenvectors
        if self.eigvecs_flag == "GOE":
            U = np.empty((d, d), rdtype, order="F")

        # Else, allocate memory for complex eigenvectors
        else:
            U = np.empty((d, d), cdtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate iid uniform random eigenvalues
            rng.random(d, rdtype, out=eigvals)
            eigvals -= 0.5
            eigvals *= std

            # Sort eigenvalues
            eigvals.sort()

            # Generate matrix from eigvecs_ensemble in place of U
            eigvecs_ens.generate_matrix(out=U)

            # Diagonalize matrix for eigenvectors
            _, U = eigh(U, overwrite_a=True, check_finite=False)

            # Yield eigenvalues and eigenvectors
            yield eigvals, U

    def eigvals_stream(self, realizs: int) -> Iterator[np.ndarray]:
        """Iterator to stream spectrum realizations."""

        # Alias random number generator
        rng = self.rng

        # Alias data type of eigenvalues
        rdtype = self.real_dtype.type

        # Alias dimension of matrix
        d = self.dim

        # Alias standard deviation of eigenvalues
        std = self.sigma

        # =================================================

        # Allocate memory for random eigenvalues
        eigvals = np.empty(d, rdtype, order="F")

        # Loop over realizations
        for _ in range(realizs):
            # Generate iid uniform random eigenvalues
            rng.random(d, rdtype, out=eigvals)
            eigvals -= 0.5
            eigvals *= std

            # Sort eigenvalues
            eigvals.sort()

            # Yield sorted eigenvalues
            yield eigvals

    def pdf(self, eigval: np.ndarray) -> np.ndarray:
        """Probability density function of the Poisson ensemble."""

        # Alias data type of eigenvalues
        rdtype = self.real_dtype.type

        # Alias ground state energy
        E0 = self.E0

        # =================================================

        # Initialize distribution with zeros
        pdf = np.zeros_like(eigval, rdtype)

        # Calculate non-zero elements
        pdf[np.abs(eigval) < E0] = 1 / 2 / E0

        # Return probability density function
        return pdf

    def cdf(self, eigval: np.ndarray) -> np.ndarray:
        """Cumulative distribution function of the Poisson ensemble."""

        # Alias data type of eigenvalues
        rdtype = self.real_dtype.type

        # Alias ground state energy
        E0 = self.E0

        # =================================================

        # Initialize distribution with zeros
        cdf = np.zeros_like(eigval, rdtype)

        # Calculate non-trivial elements
        mask = np.abs(eigval) < E0
        cdf[mask] = eigval[mask] / (2 * E0) + 0.5

        # Calculate remaining elements
        cdf[eigval > E0] = 1.0

        # Return cumulative distribution function
        return cdf
