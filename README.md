# RMTPy

**A computational laboratory for random-matrix universality—from symmetry class
and many-body Hamiltonian to spectra, scattering resonances, decay widths, time
delays, and reproducible Monte Carlo workflows.**

Random matrix theory is the art of discarding microscopic detail without losing
collective structure. RMTPy turns that idea into a Python workflow: draw a closed
Hamiltonian, resolve its spectrum, couple it to open channels, and ask which
features are model-specific and which collapse onto universal laws.

Wigner–Dyson and Bogoliubov–de Gennes ensembles, Poisson spectra, the
Sachdev–Ye–Kitaev (SYK) model, and effective non-Hermitian Hamiltonians all live
inside the same experiment. RMTPy follows their global density, local spacings,
long-range correlations, complex poles, and channel response across symmetry
classes, finite sizes, coupling strengths, and unfolding choices—then preserves
the resulting data, figures, parameters, and random state together.

## From matrix to measurement

```text
random ensemble H ──► eigenvalues/eigenvectors ──► density, spacings, form factors
      │
      └── + channel couplings W ──► H_eff, K(E), S(E), Q(E)
                                           │
                                           └──► resonances, widths, proper delays

Monte Carlo streams ──► raw + unfolded statistics ──► .npz data + .png plots
                                                          + metadata.json
```

For a closed (isolated) Hamiltonian $H$, RMTPy streams matrices, eigensystems,
or just eigenvalues. An open system—one coupled to external channels—adds
couplings $W$ and the effective Hamiltonian

$$
H_{\mathrm{eff}} = H - \frac{i}{2}WW^\dagger,
\qquad
\mathcal{E}_n = E_n - \frac{i}{2}\Gamma_n.
$$

Its complex poles provide resonance centers $E_n$ and widths $\Gamma_n$.
The reaction matrix, scattering matrix, and Wigner–Smith matrix then expose the
energy-dependent response. Proper delay times are the eigenvalues of the
Wigner–Smith matrix $Q(E)$; they measure channel dwell or response times.

## What is implemented

| Layer | Capabilities |
| --- | --- |
| Closed ensembles | GOE, GUE, GSE, Bogoliubov–de Gennes classes C and D, Poisson levels with a configurable Wigner–Dyson eigenvector basis, and the even-$q$ SYK model |
| Fermionic construction | Sparse Majorana and complex-fermion operators, charge conjugation, parity blocks, $q$-body monomials, channel-coupling matrices, and width matrices |
| Open systems | Effective non-Hermitian Hamiltonians; resonances; partial widths; reaction, scattering, and Wigner–Smith matrices; proper time delays at one or more energies |
| Density and unfolding | Empirical density estimation plus Chebyshev, Legendre, and $q$-Hermite expansions; weight, ensemble-average, and realization-specific unfolding |
| Statistics | Level and resonance densities, nearest-neighbor spacings, width and time-delay distributions, complex-energy histograms, density-coefficient distributions, spectral form factors, and connected form factors |
| Universal references | Wigner surmises, Porter–Thomas laws, connected GOE/GUE/GSE/Poisson form factors, and the compact-support proper-time-delay density |
| Experiment outputs | Streaming accumulation, normalized 1D/2D histograms, parameter-derived paths, NumPy archives, simulation metadata, and publication-oriented Matplotlib figures |

### Ensemble catalogue

The concise aliases below are exported alongside the full class names.

| Alias | Ensemble | Dyson index / defining structure | Bulk basis |
| --- | --- | --- | --- |
| `GOE` | `GaussianOrthogonalEnsemble` | $\beta=1$, real symmetric | semicircle / Chebyshev-$U$ |
| `GUE` | `GaussianUnitaryEnsemble` | $\beta=2$, complex Hermitian | semicircle / Chebyshev-$U$ |
| `GSE` | `GaussianSymplecticEnsemble` | $\beta=4$, self-dual Hermitian with Kramers degeneracy | semicircle / Chebyshev-$U$ |
| `BdGC` | `BogoliubovDeGennesCEnsemble` | particle–hole-symmetric block construction | semicircle / Chebyshev-$U$ |
| `BdGD` | `BogoliubovDeGennesDEnsemble` | imaginary antisymmetric Hermitian construction | semicircle / Chebyshev-$U$ |
| `Poisson` | `PoissonEnsemble` | $\beta=0$, independent uniform levels | uniform / Legendre |
| `SYK` | `SachdevYeKitaevEnsemble` | random $q$-body Majorana Hamiltonian; class inferred from $q$ and $N_m$ | $q$-Hermite |

All concrete ensembles are keyword-only and accept an optional NumPy-compatible
seed. In the present API, `num_majoranas` also parameterizes the classical
Gaussian, BdG, and Poisson ensembles: it must be even and satisfy
`4 <= num_majoranas <= 32`, and it sets the matrix dimension using the
many-body parity-sector convention

$$
D = 2^{N_m/2-1}.
$$

`interaction_strength`, `dtype`, `seed`, and
`max_spectral_polynomial_degree` are shared controls. SYK accepts `q` in
`{2, 4, 6, 8, 10}`; current runnable configurations require
`q < num_majoranas`. Memory-aware upper bounds on `num_majoranas` are 32 for
$q=2,4$, 26 for $q=6$, 24 for $q=8$, and 22 for $q=10$. An
`is_even_parity` flag is also present, subject to the limitation below.

### Opening a closed system

`Compound` couples an ensemble to

$$
M = \binom{N_m/2}{N_f}
$$

open channels. `num_free_complex_fermions=N_f` selects the channel count, while
`coupling_strengths` accepts either one positive scalar or a real length-$M$
nonnegative array; the default scalar is the square root of the ensemble's
spectral radius.
The object exposes effective-Hamiltonian, resonance, partial-width,
reaction-matrix, scattering-matrix, Wigner–Smith, and time-delay calculations.

Use `PoissonCompound` when the closed spectrum is `PoissonEnsemble`.
`SYKCompound` instead builds its channels from few-fermion states and a sparse
width matrix; its configured parity flag must match the parity of `N_f`, and a
symplectic ($\beta=4$) construction requires an even channel count.

## Installation

RMTPy currently runs directly from a source checkout. It has no package-build
metadata or PyPI release, so run Python from the repository root rather than
using `pip install`.

```bash
git clone https://github.com/joshua-leeman/RMTPy.git
cd RMTPy

conda create -n rmtpy-env -c conda-forge \
  python=3.11 numpy scipy matplotlib numba attrs cattrs ruff
conda activate rmtpy-env
```

The checked-in `environment.yml` supplies Python, NumPy, SciPy, and Matplotlib,
but does not yet declare the required `numba`, `attrs`, and `cattrs` packages.
The explicit environment command above is therefore the reliable setup path.

Full simulation runs save TeX-rendered figures. A working LaTeX installation
with `amsmath` and Latin Modern fonts is required for that plotting step; matrix
generation and numerical analysis do not otherwise depend on TeX.

## Quick start: closed and open spectra

```python
import numpy as np

from rmtpy.compounds import Compound
from rmtpy.ensembles import GOE

# N_m = 8 gives an 8 x 8 Hamiltonian.
ensemble = GOE(num_majoranas=8, seed=2025)

eigenvalues = next(ensemble.eigvals_stream(realizs=1))
spacings = np.diff(eigenvalues)

print(ensemble.dimension)            # 8
print(ensemble.universality_class)   # GOE
print(eigenvalues.shape, spacings.shape)

# Couple the closed system to C(4, 1) = 4 open channels.
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

print(compound.num_channels)   # 4
print(delay_times.shape)       # (number of energies, number of channels) = (1, 4)
```

Streams avoid retaining a Monte Carlo ensemble in memory. Matrix streams reuse
their working array, so copy a yielded matrix before advancing the iterator if
you need to keep it.

## Run complete statistics pipelines

Each simulation has a class API and a convenience runner exported from
`rmtpy.simulations`.

```python
from rmtpy.ensembles import GOE
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation

ensemble = GOE(
    num_majoranas=8,
    max_spectral_polynomial_degree=4,
    seed=7,
)

simulation = SpectralStatisticsSimulation(
    ensemble=ensemble,
    realizs=250,
)
simulation.run(out_dir="output")

# Results remain available in memory as typed observables.
raw_density = simulation.outputs.raw.levels.data
print(raw_density.bins)
print(raw_density.histogram)
```

| Simulation | Inputs | Accumulated products |
| --- | --- | --- |
| `SpectralStatisticsSimulation` | ensemble, realizations | level density, nearest-neighbor spacings, spectral/connected form factors, spectral-density coefficients |
| `ResonanceStatisticsSimulation` | compound, realizations | resonance centers, widths, spacings, 2D complex-energy density, resonance form factors, resonance-density coefficients |
| `PartialWidthsStatisticsSimulation` | compound, realizations, selected width indices | individual channel widths and per-state total widths, rescaled by their observed means |
| `TimeDelayStatisticsSimulation` | compound, realizations, one or more energies | raw and unfolded proper-delay histograms in a separate subdirectory for each energy |

The convenience functions are `run_spectral_statistics`,
`run_resonance_statistics`, `run_partial_widths_statistics`, and
`run_time_delay_statistics`. They call `.run()` with the default `output/` root;
use the simulation classes when you need a different destination or in-memory
access to observables.

### Why four views of one spectrum?

Raw eigenvalues mix the slowly varying bulk density with local correlations.
Unfolding removes that slow bulk-density variation by mapping a level $E$
through a CDF $F$,

$$
\widetilde E = D\,[F(E)-F(0)],
$$

so a unit interval represents one local mean spacing. RMTPy accumulates:

1. **raw** values in physical model units;
2. **weight-unfolded** values from the ensemble's leading orthogonal-polynomial
   weight (semicircle, uniform, or $q$-Hermite);
3. **average-unfolded** values from truncated ensemble-average density
   expansions of even degree `2, 4, ...`;
4. **variate-unfolded** values from a truncated density fitted separately to
   each realization.

Widths are unfolded over their finite interval,
$\widetilde\Gamma=D[F(E+\Gamma/2)-F(E-\Gamma/2)]$, and time delays are
handled through reciprocal widths. Keeping all four views makes finite-size and
non-stationary density effects visible instead of silently baking one detrending
choice into the result.

## Outputs and reproducibility

`Simulation.run()` performs realization, finalization, data persistence, and
plotting. Its output path encodes the ensemble/compound and simulation
parameters. Every run contains:

- `metadata.json` with the model, simulation arguments, and serialized RNG
  state;
- one `*_data.npz` archive per observable, containing counts/results and
  metadata;
- a matching `*_plot.png` where that observable has a plot implementation.

A representative tree is:

```text
output/
└── spectral_statistics_simulation/
    └── GOE/Nm_8/J_1p0/polydeg_4/realizs_250/
        ├── metadata.json
        ├── spectral_histogram/
        │   ├── spectral_histogram_data.npz
        │   └── spectral_histogram_plot.png
        ├── spacings_histogram_weight_unfolded/
        └── spectral_form_factors_var_unfolded_degree_4/
```

Data objects support typed `.save()`/`.load()` round trips:

```python
from rmtpy.simulations.histogram import Histogram

histogram = Histogram.load("path/to/spectral_histogram_data.npz")
```

Only load `.npz` files you trust: metadata restoration uses NumPy object arrays
and therefore enables pickle loading.

Ensembles and compounds also support dictionary conversion through
`RMT_CONVERTER`, including polymorphic ensemble reconstruction. A seed and saved
RNG state improve reproducibility, but bitwise identity across NumPy, SciPy,
Numba, or BLAS/LAPACK versions is not promised.

## Numerical design and limits

- Matrix construction and polynomial recurrences use Numba-compiled kernels;
  eigensystems use SciPy BLAS/LAPACK; fermion operators are assembled sparsely.
  The first call to a compiled kernel includes one-time JIT overhead.
- Monte Carlo APIs stream realizations, and form factors evaluate time grids in
  chunks to limit temporary memory.
- Average-polynomial unfolding lazily estimates its reference coefficients from
  `max(8192 // dimension, 10)` additional ensemble draws; budget those separately
  from the simulation's requested `realizs`.
- Dense diagonalization still scales cubically in `dimension`, while the
  many-body Hilbert space grows exponentially in `num_majoranas`. Begin with
  modest sizes; an accepted constructor value is not a promise that a dense run
  fits your machine.
- Histogram values outside an observable's fixed support are omitted, and CDF
  interpolators extrapolate beyond their construction grids; inspect supports
  when studying tails.
- **Known limitation:** despite its name and iterator annotation,
  `Compound.scattering_matrix_stream` is not a generator; it returns one tuple
  after the first underlying realization. Use it only with `realizs=1`; the
  reaction, resonance, and Wigner–Smith methods provide true streams.
- SYK's `is_even_parity` flag is validated by `SYKCompound` but is not currently
  propagated into `MajoranaFermionBasis`; matrix construction therefore remains
  in the basis's default even-parity block.
- `PoissonEnsemble.porter_thomas_distribution()` currently has a call-signature
  defect. The general reference law in `rmtpy.universal` remains available, but
  the Poisson-specific wrapper/theory overlay should not yet be relied upon.
- Saved plots are produced correctly as part of `Simulation.run()`, but the
  standalone `plot_data(path)` registry dispatch is not currently reliable for
  generic histogram archives.
- The public interface is research-stage and source-first: there is no semantic
  version, packaged release, CI workflow, or exhaustive numerical validation
  yet.

## Repository map

```text
rmtpy/
├── ensembles/       # closed random-matrix and SYK ensembles
├── compounds/       # channel coupling and open-system observables
├── simulations/     # Monte Carlo outputs, unfolding, persistence, and plots
├── density.py       # density models, CDFs, and unfolding primitives
├── fermions.py      # sparse Majorana/complex-fermion algebra
├── polynomials.py   # Chebyshev, Legendre, and q-Hermite systems
├── universal.py     # analytical universal reference laws
└── conversion.py    # attrs/cattrs serialization and path/label helpers

tests/
└── test_smoke.py    # compact integration coverage
```

## Tests and style

From the repository root:

```bash
python -m unittest discover -s tests -v
ruff check rmtpy tests
```

The current smoke suite verifies subtype-aware ensemble serialization,
histogram `.npz` round trips, normalization of the analytical time-delay law,
nonnegative multi-channel proper delays up to numerical tolerance, and
construction/metadata of all four simulation families. It is integration smoke
coverage, not a comprehensive proof of every ensemble law or plotting path.

## License

RMTPy is available under the [MIT License](LICENSE). Copyright © 2025 Joshua
Leeman.
