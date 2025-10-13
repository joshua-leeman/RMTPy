# rmtpy/ensembles/base/ensemble.py

# Postponed evaluation of annotations
from __future__ import annotations

# Standard library imports
import ast
import inspect
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from attrs import asdict, field, fields_dict, frozen
from attrs.validators import gt


# -------------------------------
# Random Matrix Ensemble Registry
# -------------------------------
ENSEMBLE_REGISTRY: dict[str, type[Ensemble]] = {}


# ------------------------------------
# Random Matrix Ensemble Class Factory
# ------------------------------------
def rmt_ensemble(**kwargs: Any) -> Ensemble:
    """Create a random matrix ensemble instance by name."""

    # Call ensemble constructor with provided keyword arguments as dictionary
    return Ensemble.create(kwargs)


# ---------------------------------
# Random Matrix Ensemble Base Class
# ---------------------------------
@frozen(kw_only=True, eq=False, weakref_slot=False, getstate_setstate=False)
class Ensemble(ABC):

    # Random number generator seed
    seed: Any = field(default=None)

    # Complex data type of matrix entries
    dtype: np.dtype = field(default=np.dtype("complex64"), converter=np.dtype)

    # Data type for real and imaginary parts of matrix entries
    real_dtype: np.dtype = field(init=False, repr=False)

    # Size of generated matrices
    dim: int = field(
        converter=int, validator=gt(0), metadata={"dir_name": "D", "latex_name": "D"}
    )

    # Random number generator
    rng: np.random.Generator = field(init=False, repr=False)

    # Set real data type based on complex data type
    @real_dtype.default
    def __real_dtype_default(self) -> np.dtype:
        """Set the real data type based on the complex data type."""

        # If dtype is complex, return the real part's dtype
        return np.dtype(self.dtype.char.lower())

    # Initialize random number generator with seed
    @rng.default
    def __rng_default(self) -> np.random.Generator:
        """Initialize the ensemble's random number generator."""

        # Validate seed and create random number generator, raise error if invalid
        try:
            # If seed is a string, attempt literal evaluation
            if isinstance(self.seed, str):
                # Safely evaluate string and store seed
                object.__setattr__(self, "seed", ast.literal_eval(self.seed))

            # If seed is a dictionary, assume it is a bit generator state
            if isinstance(self.seed, dict):
                # Determine BitGenerator type from dictionary
                bit_gen_cls = getattr(np.random, self.seed["bit_generator"])

                # Create BitGenerator instance from state
                bit_gen: np.random.BitGenerator = bit_gen_cls()

                # Set state of BitGenerator using provided state
                bit_gen.state = self.seed

                # Create and return random number generator
                return np.random.default_rng(bit_gen)

            # Else, normalize seed and create random number generator
            else:
                # Create random number generator with provided seed
                rng = np.random.default_rng(self.seed)

                # Store seed as BitGenerator state
                object.__setattr__(self, "seed", rng.bit_generator.state)

                # Return random number generator
                return rng

        # Handle exceptions related to raw string evaluation of seed
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid seed value: {self.seed}--{e}") from e

        # Handle exceptions related to BitGenerator state
        except (AttributeError, KeyError) as e:
            raise AttributeError(f"Invalid BitGenerator state {self.seed}--{e}") from e

        # Handle TypeError for unsupported seed types
        except TypeError as e:
            raise TypeError(
                f"Invalid seed type: {type(self.seed).__name__}. "
                "Expected one of: None, int, Sequence[int], "
                "dict, np.integer, NDArray[np.integer], "
                "SeedSequence, BitGenerator, Generator, or RandomState. "
            ) from e

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        """Register concrete subclasses in the ensemble registry."""

        # Include only concrete classes in registry
        if not inspect.isabstract(cls):
            # Convert ensemble class to registry key format
            ens_key = re.sub(r"_", "", cls.__name__).lower()

            # Normalize class name to registry key format
            ENSEMBLE_REGISTRY[ens_key] = cls

    @classmethod
    def create(cls, src: dict[str, Any] | Ensemble) -> Ensemble:
        """Create an instance of the ensemble with given parameters."""

        # Import module-level converter
        from ...serialization import converter

        # Convert dictionary to ensemble instance
        return converter.structure(src, cls)

    @property
    def matrix_memory(self) -> int:
        """Calculate memory used to store a generated matrix in bytes."""

        # Calculate memory used to store a generated matrix
        return self.dtype.itemsize * self.dim**2

    @property
    def rng_state(self) -> dict[str, Any]:
        """Get current state of the random number generator."""

        # Return state of random number generator
        return self.rng.bit_generator.state

    @property
    def to_dir(self) -> Path:
        """Generate directory Path used for storing data related to the RMT instance."""

        # Begin path with class name
        dir_path = Path(self._dir_name)

        # Loop through fields of class instance
        for name, attr in fields_dict(type(self)).items():
            # Use only fields labeled for path inclusion
            if attr.metadata.get("dir_name", None) is not None:
                # Sanitize field value for directory representation
                val = re.sub(r"[^\w\-.]", "_", str(asdict(self)[name]))

                # Append string representation of field value to path
                dir_path /= f"{attr.metadata['dir_name']}_{val.replace('.', 'p')}"

        # Return path
        return dir_path

    @property
    def to_latex(self) -> str:
        """Generate LaTeX representation of the RMT instance."""

        # Begin LaTeX string with ensemble's latex name
        latex_str = f"${self._latex_name}"

        # Loop through fields of class instance
        for name, attr in fields_dict(type(self)).items():
            # Use only fields labeled for LaTeX inclusion
            if attr.metadata.get("latex_name", None) is not None:
                # Append LaTeX formatted field value to string
                latex_str += rf"\ {attr.metadata['latex_name']}={asdict(self)[name]}"

        # Close LaTeX string and return it
        return latex_str + "$"

    @property
    def _dir_name(self) -> str:
        """Generate directory name used for storing Ensemble instance data."""

        # Insert underscores between words and acronyms in class name
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", type(self).__name__)
        return re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)

    @property
    def _latex_name(self) -> str:
        """Generate LaTeX representation of the ensemble class name."""

        # Use class name as LaTeX name, replacing underscores with spaces
        return f"\\textrm{{{re.sub(r'_', ' ', type(self).__name__)}}}"

    def set_rng_state(self, state: dict[str, Any] | None) -> None:
        """Set the state of the random number generator."""

        # Check if state is a dictionary
        if isinstance(state, dict):
            # Try to set state of random number generator
            try:
                self.rng.bit_generator.state = state

            # Handle exceptions related to setting BitGenerator state
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid RNG state: {e}") from e

        # Check if state is not None
        elif state is not None:
            # Raise error if state is not a dictionary or None
            raise TypeError(
                f"Expected RNG state to be a dictionary or None, "
                f"got {type(state).__name__} instead."
            )

    @abstractmethod
    def generate(self, offset: np.ndarray | None = None) -> np.ndarray:
        """Generate a random matrix."""

        # This method should be implemented by subclasses
        pass
