from typing import TypeAlias

from .base import RandomMatrixEnsemble
from .bdgc import BogoliubovDeGennesCEnsemble
from .bdgd import BogoliubovDeGennesDEnsemble
from .goe import GaussianOrthogonalEnsemble
from .gse import GaussianSymplecticEnsemble
from .gue import GaussianUnitaryEnsemble
from .many_body import ManyBodyEnsemble
from .poisson import PoissonEnsemble
from .syk import SachdevYeKitaevEnsemble
from .wigner_dyson import WignerDysonEnsemble

RME = RandomMatrixEnsemble
BdGC = BogoliubovDeGennesCEnsemble
BdGD = BogoliubovDeGennesDEnsemble
GOE = GaussianOrthogonalEnsemble
GSE = GaussianSymplecticEnsemble
GUE = GaussianUnitaryEnsemble
MBE = ManyBodyEnsemble
Poisson = PoissonEnsemble
SYK = SachdevYeKitaevEnsemble
WDE = WignerDysonEnsemble

EnsembleLike: TypeAlias = (
    ManyBodyEnsemble | WignerDysonEnsemble | PoissonEnsemble | SachdevYeKitaevEnsemble
)

__all__ = [
    "RandomMatrixEnsemble",
    "BogoliubovDeGennesCEnsemble",
    "BogoliubovDeGennesDEnsemble",
    "GaussianOrthogonalEnsemble",
    "GaussianSymplecticEnsemble",
    "GaussianUnitaryEnsemble",
    "ManyBodyEnsemble",
    "PoissonEnsemble",
    "SachdevYeKitaevEnsemble",
    "WignerDysonEnsemble",
    "RME",
    "BdGC",
    "BdGD",
    "GOE",
    "GSE",
    "GUE",
    "MBE",
    "Poisson",
    "SYK",
    "WDE",
]
