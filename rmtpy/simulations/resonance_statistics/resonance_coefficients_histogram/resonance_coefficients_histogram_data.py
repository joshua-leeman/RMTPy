from typing import ClassVar

import attrs

from ...statistics import CoefficientsHistogram


@attrs.frozen(kw_only=True, eq=False, weakref_slot=False)
class ResonanceCoefficientsHistogram(CoefficientsHistogram):
    coefficient_name: ClassVar[str] = "resonance"
