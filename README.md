# RMTPy

**A computational laboratory for random-matrix universality: from a symmetry
class or many-body Hamiltonian to spectra, scattering resonances, decay widths,
proper time delays, and reproducible Monte Carlo experiments.**

Random matrix theory discards microscopic detail without discarding collective
structure. RMTPy turns that idea into a Python workflow: draw a closed
Hamiltonian, resolve its spectrum, couple it to open channels, and compare raw
and unfolded observables with universal laws.

The repository is deliberately organized around the physics. The final object
model is not flat, but it is small: an `Ensemble` describes a closed system, a
`Compound` opens it, a `Simulation` runs an experiment, and an `Observable`
holds one numerical result. Density models and fermion bases remain separate
objects because they cache expensive, reusable mathematics. Plot objects are
temporary views and are not retained by a simulation.

```text
closed Ensemble H ──► eigvals/eigvecs ──► DensityModel ──► spectral statistics
        │
        └── Compound(H, channels) ──► H_eff, K(E), S(E), Q(E)
                                             │
                                             └──► resonances, widths, delays

Simulation ──► typed Outputs ──► Observable ──► mutable Data accumulator
                                      │
                                      └──► transient Plot at save time
```

For an open system,

$$
H_{\mathrm{eff}}=H-\frac{i}{2}WW^\dagger,
\qquad
\mathcal{E}_n=E_n-\frac{i}{2}\Gamma_n.
$$

The poles $\mathcal{E}_n$ give resonance centers $E_n$ and widths $\Gamma_n$.
The reaction matrix, scattering matrix, and Wigner--Smith matrix describe the
energy-dependent channel response; the eigenvalues of the Wigner--Smith matrix
$Q(E)$ are the proper delay times.

## Installation

RMTPy currently runs from a source checkout. It has no package-build metadata or
PyPI release, so run Python from the repository root.

```bash
git clone https://github.com/joshua-leeman/RMTPy.git
cd RMTPy
conda env create -f environment.yml
conda activate rmtpy-env
```

The environment includes Python 3.11, NumPy, SciPy, Matplotlib, Numba, `attrs`,
`cattrs`, and Ruff. Numerical work does not require TeX. Saving the
publication-oriented plots does require a working LaTeX installation with
`amsmath` and Latin Modern fonts.

## Quick start

### A closed spectrum and an open system

```python
import numpy as np

from rmtpy.compounds import Compound
from rmtpy.ensembles import GOE

# N_m = 8 gives one 8 x 8 parity block.
ensemble = GOE(num_majoranas=8, seed=2025)

eigenvalues = next(ensemble.eigvals_stream(realizs=1))
spacings = np.diff(eigenvalues)

print(ensemble.dimension)           # 8
print(ensemble.universality_class)  # GOE

# N_f = 1 gives C(4, 1) = 4 open channels.
compound = Compound(
    ensemble=ensemble,
    num_free_complex_fermions=1,
)

resonances = next(compound.resonances_stream(realizs=1))
delay_times, closed_levels = next(
    compound.time_delays_stream(
        realizs=1,
        energies=np.array([0.0]),
    )
)

centers = resonances.real
widths = -2.0 * resonances.imag

print(compound.num_channels)  # 4
print(delay_times.shape)      # (number of energies, channels) = (1, 4)
```

Monte Carlo methods are streams so that realizations do not accumulate in
memory. Matrix streams reuse their working array. Copy a yielded matrix before
advancing the iterator if it must be retained.

### A complete spectral-statistics experiment

```python
from rmtpy.ensembles import GOE
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation

simulation = SpectralStatisticsSimulation(
    ensemble=GOE(
        num_majoranas=8,
        max_spectral_polynomial_degree=4,
        seed=7,
    ),
    realizs=250,
)
simulation.run(out_dir="output")

# A short public lookup avoids a long output path.
raw_density = simulation.get_data("spectral_histogram")
print(raw_density.bins)
print(raw_density.histogram)

# The typed hierarchy remains available when its structure is useful.
assert raw_density is simulation.outputs.raw.levels.data

# Metadata can select a related group. This returns three observables:
# levels, spacings, and form factors at degree 4.
degree_four = simulation.find_observables(unfolding="avg", degree=4)
```

`get_data()` and `get_observable()` require exactly one match and raise a
`LookupError` if a selection is absent or ambiguous. `find_observables()` always
returns a tuple. A file name may be given with or without its trailing `_data`.

## The objects that matter

The hierarchy is easiest to read as a sequence of scientific responsibilities.

| Object | Owns | Why it is a separate object |
| --- | --- | --- |
| `RandomMatrixEnsemble` / `ManyBodyEnsemble` | Configuration, RNG, matrix/eigensystem streams, symmetry class | This is the closed Hamiltonian family. It is the natural source of universal reference laws. |
| `DensityModel` | Support, weight, orthogonal polynomials, lazy PDF/CDF and coefficient caches | Density estimation and unfolding are reused by direct calculations, simulations, and plots. The object avoids recomputing expensive calibration data. |
| `MajoranaFermionBasis` | Sparse Majorana and complex-fermion operators, parity slice, vacuum, charge conjugation | These arrays are expensive algebraic infrastructure shared by SYK matrix and channel construction. They are created lazily. |
| `Compound` | One ensemble, channel strengths, resonance density, open-system streams | It represents a different physical system: the closed Hamiltonian plus its coupling to external channels. |
| `Simulation` | Immutable experiment inputs, metadata, lifecycle, one output bundle | It coordinates realization, finalization, persistence, and plotting. It does not implement the physics already owned by an ensemble or compound. |
| Output bundle | Typed grouping by quantity, unfolding, degree, and sometimes energy | It preserves scientifically meaningful alignment. It contains references to observables, not copies of the ensemble or compound. |
| `Observable` | One `Data` object, optional finalizer, optional plot views | It is a thin descriptor. It does not retain a `Plot`. Multiple transient views may share one persisted result without duplicating its arrays. |
| `Data` | Mutable NumPy buffers and result metadata | Frozen `attrs` wiring prevents accidental reassignment while arrays remain mutable for streaming accumulation. |
| `Plot` | A temporary view of one `Data` object | It is constructed only while saving. All plots in one save pass share one detached model reconstructed from initial metadata, so the live simulation and RNG remain untouched. |

The answer to “is the repository too object-oriented?” is therefore: the
physics hierarchy is justified; the old orchestration was too implicit. The
current code keeps the physical objects and typed output grid while removing
simulation deserialization magic, recursive object discovery, retained plots,
plot registries, one-field legend subclasses, and forwarding factory chains.

### How `attrs` construction works here

All principal configuration objects are frozen, keyword-only `attrs` classes.
Fields appear in this order:

1. required physical inputs;
2. optional controls with converters and validators;
3. derived public attributes with `init=False` factories;
4. private lazy caches prefixed with `_`.

An ensemble first validates its physical inputs, derives quantities such as
dimension and spectral radius, and then attaches one `DensityModel`. A
`Compound` converts or accepts a live ensemble, derives the channel count,
normalizes a copied read-only coupling array, and attaches a resonance-density
model. SYK additionally derives one `MajoranaFermionBasis`; its expensive sparse
monomials remain lazy.

A simulation constructs its output bundle once through an `attrs.Factory` that
receives the completed simulation. The bundle explicitly implements
`iter_observables()`, so saving order is visible in the source rather than
discovered by recursive reflection. Base metadata is generated automatically
from the simulation's `init=True` fields.

## Implemented physics

### Ensemble catalogue

The concise aliases below are exported with the full class names.

| Alias | Ensemble | Dyson index / structure | Bulk basis |
| --- | --- | --- | --- |
| `GOE` | `GaussianOrthogonalEnsemble` | $\beta=1$, real symmetric | semicircle / Chebyshev-$U$ |
| `GUE` | `GaussianUnitaryEnsemble` | $\beta=2$, complex Hermitian | semicircle / Chebyshev-$U$ |
| `GSE` | `GaussianSymplecticEnsemble` | $\beta=4$, self-dual Hermitian with Kramers degeneracy | semicircle / Chebyshev-$U$ |
| `BdGC` | `BogoliubovDeGennesCEnsemble` | particle--hole-symmetric block construction | semicircle / Chebyshev-$U$ |
| `BdGD` | `BogoliubovDeGennesDEnsemble` | imaginary antisymmetric Hermitian construction | semicircle / Chebyshev-$U$ |
| `Poisson` | `PoissonEnsemble` | independent uniform levels with configurable GOE/GUE/GSE eigenvectors | uniform / Legendre |
| `SYK` | `SachdevYeKitaevEnsemble` | random even-$q$ Majorana Hamiltonian; class inferred from $q,N_m$ | $q$-Hermite |

The Gaussian, BdG, Poisson, and SYK implementations share the many-body parity
sector convention

$$
D=2^{N_m/2-1}.
$$

`num_majoranas` must be even and satisfy `4 <= num_majoranas <= 32`.
`interaction_strength`, `dtype`, `seed`, and
`max_spectral_polynomial_degree` are common controls. SYK accepts `q` in
`{2, 4, 6, 8, 10}` with `q < num_majoranas`; its memory-aware upper limits on
$N_m$ are 32 for $q=2,4$, 26 for $q=6$, 24 for $q=8$, and 22 for $q=10$.
`is_even_parity` selects the actual parity block and appears in SYK labels and
output paths.

`PoissonEnsemble.eigvecs_ensemble_flag` chooses `GOE`, `GUE`, or `GSE`
eigenvectors. The nested eigenvector ensemble intentionally shares the one
Poisson RNG; its choice is encoded in the output path.

### Opening a closed system

`Compound` couples an ensemble to

$$
C=\binom{N_m/2}{N_f}
$$

channels. `num_free_complex_fermions=N_f` selects the channel count, which may
not exceed the ensemble dimension. `coupling_strengths` accepts one positive
finite scalar or a real, finite, nonnegative array of length $C$. Input arrays
are copied and made read-only. The default scalar is the square root of the
spectral radius.

The object exposes effective-Hamiltonian, resonance, partial-width, reaction,
scattering, Wigner--Smith, and proper-delay streams. Use `PoissonCompound` for a
Poisson closed spectrum. `SYKCompound` builds its couplings from few-fermion
states and a sparse width matrix; the parity of $N_f$ must match the configured
SYK parity, and a symplectic construction requires equal strengths within each
Kramers pair.

### Universal references

`rmtpy.universal` supplies Wigner surmises, Porter--Thomas laws, connected
GOE/GUE/GSE/Poisson spectral form factors, eigenvalue degeneracies, symmetry
labels, and the compact-support proper-time-delay density. Ensemble and compound
methods bind the relevant dimension, Dyson index, or channel count.

## Simulation families and output layout

Each family has a class API and a convenience runner exported from
`rmtpy.simulations`.

| Simulation | Inputs | Main products |
| --- | --- | --- |
| `CDOEvolutionSimulation` | ensemble, realizations, initial state, time grid | basis probabilities, classical and quantum purity, von Neumann entropy, ensemble-averaged KL divergence |
| `SpectralStatisticsSimulation` | ensemble, realizations | levels, nearest-neighbor spacings, spectral and connected form factors, density coefficients |
| `ResonanceStatisticsSimulation` | compound, realizations | centers, widths, spacings, 2-D complex-energy density, resonance form factors, density coefficients |
| `PartialWidthsStatisticsSimulation` | compound, realizations, width selections | selected channel widths and per-state total widths, scaled by observed means |
| `TimeDelayStatisticsSimulation` | compound, realizations, probe energies | raw and unfolded proper-delay histograms for each energy |
| `TransmissionCoefficientsSimulation` | compound, realizations, channel selections | energy-resolved channel transmission coefficients and the Weisskopf estimate |

Let `M=max_spectral_polynomial_degree`, let
`T=(2, 4, ..., <= M)`, and let `k=len(T)`.

| Output bundle | Observable count | Typed organization |
| --- | ---: | --- |
| CDO evolution | `1` (or `2`) | one time-resolved dynamics observable; optional retained evolved states |
| spectral | `M + 6 + 6*k` | coefficients; raw and weight groups; average and variate groups by degree |
| resonance | `M + 10 + 10*k` | coefficients; five-quantity raw and weight groups; five-quantity average and variate groups by degree |
| time delay at `n` energies | `2*n*(k + 1)` | raw/weight by energy; average/variate by degree, then energy |
| partial width | `len(width_indices)` | one histogram per requested selection, in request order |
| transmission coefficients for `c` channels | `c + 1` | one `T_a(E)` plot per requested channel and one all-channel Weisskopf-estimate plot |

Degree zero is a supported, useful small-run configuration. It produces raw and
weight-unfolded outputs but no coefficient, average-unfolded, or
variate-unfolded groups. Degree one adds a degree-one coefficient histogram but
still has no even-degree truncated groups.

For partial widths, `(state, channel)` selects one channel contribution and
`(state,)` sums all channels for that state. Selections must be nonempty,
unique, length one or two, and in bounds. Defaults are filtered to the actual
matrix and channel dimensions. File names are explicit, for example
`partial_width_state_1_channel_0_histogram` and
`total_width_state_1_histogram`.

Time-delay energies may be a scalar or a finite, nonempty one-dimensional
array-like. They are copied into a contiguous, read-only `float64` array.
Numeric duplicates (including signed zero) and distinct values that would
collide after five-significant-digit output-path formatting are rejected.

Transmission coefficients use the same `Compound` coupling strengths as the
other open-system experiments. Select the channel labels `a` with
`channel_indices`. The simulation evaluates every selected channel on an
energy grid spanning `compound.ensemble.spectral_density.plot_range`, the same
horizontal range used by the raw spectral-density plot. At every grid point it
averages the diagonal scattering amplitudes over realizations before calculating

$$
T_a(E)=1-\left|\left\langle S_{aa}(E)\right\rangle\right|^2.
$$

This is deliberately different from averaging `1 - |S_aa|^2` realization by
realization. The former is the channel transmission coefficient. Each requested
channel gets its own `transmission_coefficients_plot.png`, with energy on the
horizontal axis and outputs grouped under `channel_<a>`.

The simulation also produces exactly one Weisskopf estimate,

$$
\Gamma_{\mathrm{Weisskopf}}(E)
=\frac{d(E)}{2}\sum_{a=1}^{\Lambda}T_a(E),
\qquad
d(E)=\frac{1}{D\,\rho_{\mathrm{weight}}(E)}.
$$

Unlike the individually requested transmission-coefficient outputs, this sum
always includes all `compound.num_channels` open channels. Its plot uses the
same energy grid and horizontal limits as every $T_a(E)$ plot. Where the
spectral weight density vanishes outside its physical support, the estimate is
stored as undefined rather than dividing by zero.

```python
from rmtpy.compounds import Compound
from rmtpy.ensembles import GOE
from rmtpy.simulations.transmission_coefficients_simulation import (
    TransmissionCoefficientsSimulation,
)

compound = Compound(
    ensemble=GOE(num_majoranas=8, seed=7),
    coupling_strengths=[0.6, 1.0, 1.4, 1.8],
)
simulation = TransmissionCoefficientsSimulation(
    compound=compound,
    realizs=500,
    channel_indices=(0, 2, 3),
)
simulation.run(out_dir="output")
```

`CDOEvolutionSimulation` evolves one normalized initial state under each closed
Hamiltonian realization and forms the chaotic density operator

$$
\rho_{\mathrm{CDO}}(t)=\frac{1}{R}\sum_{r=1}^{R}
|\psi_r(t)\rangle\langle\psi_r(t)|.
$$

Its time grid is zero followed by points spaced logarithmically in base $D$ and
scaled by $j_{1,1}/E_0$. The KL output is the realization average
$R^{-1}\sum_r D_{\mathrm{KL}}(\bar p\Vert p_r)$ for basis probabilities, where
$\bar p=R^{-1}\sum_r p_r$. This reverse divergence is infinite if an individual
realization assigns zero probability where $\bar p$ is positive. The accumulator
chooses between state factors and a packed Hermitian density operator according
to which exact representation is smaller; evolved states are therefore not
retained as a persisted output unless `retain_evolved_states=True` is requested.

The one CDO dynamics result produces three transient views:
`cdo_probabilities_plot`, `cdo_purities_plot`, and `cdo_information_plot`.
All three reuse the raw spectral-form-factor time axis exactly, including its
$u=t/t_0$ label, dimension-scaled ticks, and plotted time range. Probabilities
and purities use the same base-$D$ logarithmic vertical range as the form
factor. Entropy and KL divergence use a linear vertical range from $0$ to
$\log D$, matching the established information-plot scale.

## Unfolding

Raw levels mix slowly varying bulk density with local correlations. Given a CDF
$F$, RMTPy unfolds a value through

$$
\widetilde E=D\,[F(E)-F(0)]
$$

and a width through its finite interval,

$$
\widetilde\Gamma
=D\,[F(E+\Gamma/2)-F(E-\Gamma/2)].
$$

The simulation grids retain four scientifically distinct views:

1. **raw** values in model units;
2. **weight-unfolded** values from the leading ensemble weight (semicircle,
   uniform, or $q$-Hermite);
3. **average-unfolded** values from ensemble-average density expansions at
   truncation degrees in `T`;
4. **variate-unfolded** values from coefficients fitted to each realization.

Spectral unfolding uses `ensemble.spectral_density`. Resonance unfolding uses
`compound.resonance_density` and fits variate coefficients to the resonance
centers. Time-delay unfolding deliberately uses the closed spectral density:
it maps $\tau\mapsto1/\tau$, unfolds that width about the fixed probe energy,
and then takes the reciprocal. Nonfinite and nonpositive values are omitted.

One mixed representation is intentional. In every “unfolded” 2-D
complex-energy histogram, the horizontal coordinate remains the physical
$E/E_0$, while only the vertical width is unfolded. The separate resonance
histograms, spacings, and form factors do use unfolded centers. This lets a
reader see unfolded-width variation across physical energy.

## Persistence, paths, and plotting

`Simulation.run()` realizes the Monte Carlo stream, finalizes all observables,
saves their data, and then saves plots. Output paths are derived only from attrs
fields carrying `dir_name` metadata.

```text
output/
└── spectral_statistics_simulation/
    └── GOE/Nm_8/J_1p0/max_polydeg_4/realizs_250/
        ├── metadata.json
        ├── spectral_histogram/
        │   ├── spectral_histogram_data.npz
        │   └── spectral_histogram_plot.png
        ├── spacings_histogram_weight_unfolded/
        └── spectral_form_factors_var_unfolded_degree_4/
```

Compound paths add the compound type, `Nf`, and either a constant coupling value
or a stable hash of a nonconstant coupling array. Poisson paths add the
eigenvector symmetry class; SYK paths add `q` and parity. Time-delay observables
are routed beneath `energy_<value>` directories, where the energy uses five
significant digits and replaces `-` by `n` and `.` by `p`, such as
`energy_n0p25`.

Every run writes one root `metadata.json`. Every observable writes
`<name>/<name>_data.npz` and, when plot views exist, one or more PNG files in
the same `<name>/` directory. Most observables use the conventional
`<name>_plot.png` file name; multi-view data such as CDO dynamics use explicit
quantity names. The archive metadata contains the initial simulation arguments
and RNG state.

Data objects support typed save/load round trips:

```python
from rmtpy.simulations.histogram import Histogram

histogram = Histogram.load("path/to/spectral_histogram_data.npz")
```

Only load archives you trust. Metadata is stored as an object array, so loading
uses NumPy's pickle support. New archives contain only attrs fields; there is no
spurious `allow_pickle` archive member.

Whole simulation deserialization is intentionally not part of the API. The old
registry path was incomplete and attempted to discover arbitrary output fields.
Ensembles and compounds still support polymorphic `RMT_CONVERTER` dictionary
round trips, and individual `Data` objects remain loadable.

Plots created by `Simulation.run()` share one transient detached model instead
of reconstructing one model per observable. The detached graph is released when
plot saving finishes, and plotting cannot advance the live simulation RNG. To
replot a saved archive independently, provide the view explicitly:

```python
from rmtpy.simulations.plot import plot_data
from rmtpy.simulations.spectral_statistics.spectral_histogram import (
    SpectralHistogramPlot,
)

plot_data(
    "path/to/spectral_histogram_data.npz",
    plot_cls=SpectralHistogramPlot,
)
```

## Source organization and coding conventions

The code follows one consistent local pattern:

- imports are ordered as future, standard library, third party, absolute
  `rmtpy`, then relative package imports;
- typed uppercase constants and their attrs metadata live near the top of a
  module;
- small `compute_*`, `create_*`, `normalize_*`, and validation helpers appear
  before the class whose fields use them;
- classes use explicit physics names, keyword-only construction, converters,
  and validators rather than large generic configuration dictionaries;
- abbreviations such as `eigvals`, `eigvecs`, `realizs`, `num_pts`, and
  `std_dev` are used consistently in numerical paths;
- feature subpackages keep observables, typed outputs, plots, and the simulation
  coordinator near one another;
- concise aliases (`GOE`, `GUE`, `SYK`, and so on) are exported from package
  `__init__.py` files while implementations retain their full names.

```text
rmtpy/
├── ensembles/       # closed Gaussian, BdG, Poisson, and SYK Hamiltonians
├── compounds/       # channel coupling and open-system matrix streams
├── simulations/     # experiments, output bundles, accumulators, and plots
├── density.py       # density models, interpolated PDFs/CDFs, unfolding support
├── fermions.py      # sparse Majorana and complex-fermion algebra
├── polynomials.py   # Chebyshev, Legendre, and q-Hermite systems
├── universal.py     # analytical universal reference laws
├── conversion.py    # attrs/cattrs conversion plus path and label helpers
└── validators.py    # shared numerical validators

tests/
├── test_smoke.py
├── test_numerical_regressions.py
├── test_simulation_outputs.py
└── test_simulation_lifecycle.py
```

The local ignored `notebooks/` and `archive/` directories are research history,
not runtime dependencies of `rmtpy`.

## Numerical behavior and limits

- Matrix construction and polynomial recurrences use Numba; eigensystems use
  SciPy BLAS/LAPACK; fermion operators are assembled sparsely. The first call to
  a compiled kernel includes JIT overhead.
- Dense diagonalization scales cubically in `dimension`, while the many-body
  Hilbert space grows exponentially in `num_majoranas`. Begin with modest sizes.
- Average-polynomial unfolding lazily estimates reference coefficients from
  `max(8192 // dimension, 10)` additional ensemble draws. The result is cached,
  but those initial calibration draws advance the same RNG. Run order therefore
  matters when density caches are first populated.
- Histograms use fixed half-open supports and omit values outside them. CDF
  interpolators extrapolate beyond their construction grids; inspect supports
  when studying tails.
- Zero-count histograms and form factors finalize to finite zeros. Partial-width
  normalization instead requires a positive finite observed mean and raises
  before changing any bin edges if that condition fails.
- Output roots do not encode every input: seed, dtype, the time-energy list, and
  partial-width selections are absent from the root path. Reusing a destination
  can overwrite matching files and does not remove stale files from an earlier
  run. Time energies are safe within one simulation but values from separate
  runs can still share the same five-digit directory name.
- A saved seed and RNG state support replay in the same software stack, but
  bitwise identity across NumPy, SciPy, Numba, or BLAS/LAPACK versions is not
  promised.
- This remains a source-first research codebase: there is no semantic version,
  packaged release, CI workflow, or exhaustive proof of every finite-size law.

## Tests and style

From the repository root:

```bash
python -m unittest discover -s tests -v
ruff check rmtpy tests
ruff format --check rmtpy tests
```

The suites cover polymorphic conversion and data round trips, analytical and
matrix-level numerical regressions, all four output hierarchies, degree-zero
experiments, partial-width validation and normalization, explicit observable
lookup, transient plotting, path collisions, and finite empty-result handling.

## License

RMTPy is available under the [MIT License](LICENSE). Copyright © 2025 Joshua
Leeman.
