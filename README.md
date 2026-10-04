# RMTPy

RMTPy is a Python codebase for numerical studies of Sachdev–Ye–Kitaev (SYK)
many-body Hamiltonians, classical random-matrix ensembles, and open quantum
systems. It generates closed Hamiltonians, couples them to decay channels, and
accumulates Monte Carlo statistics for spectra, scattering poles, decay widths,
proper delay times, and transmission coefficients.

A calculation has three main layers. An ensemble supplies one closed random
Hamiltonian realization. A compound couples open channels to that Hamiltonian. A
simulation repeats the calculation and combines the resulting measurements.

```text
closed system
    Ensemble ──► H ──► eigenvalues and eigenvectors
        │                       │
 + open channels                └──► densities, spacings, form factors
        │
        └──► CompoundEnsemble ──► H_eff, K(E), S(E), Q(E)
                                      │
                                      └──► poles, widths, delays, transmissions

physical source ──► Simulation ──► finalized Data ──► NPZ + manifest + plots
```

| Part of the calculation | Implemented capabilities |
| --- | --- |
| Closed ensembles | The Sachdev–Ye–Kitaev model, GOE, GUE, GSE, Bogoliubov–de Gennes classes C and D, and Poisson spectra |
| Fermionic construction | Sparse Majorana and complex-fermion operators, parity blocks, charge conjugation, few-fermion states, and decomposed $`q`$-body monomials |
| Open systems | Effective non-Hermitian Hamiltonians, resonances, partial widths, reaction and scattering matrices, Wigner–Smith matrices, and proper delay times |
| Statistical experiments | Spectral, resonance, partial-width, time-delay, and transmission-coefficient simulations |
| Density analysis | Semicircle, uniform, and $`q`$-Hermite weights; polynomial density expansions; empirical densities; and four forms of unfolding |
| Analytical references | Wigner surmises, Porter–Thomas laws, connected form factors, eigenvalue degeneracies, and a proper-delay density |
| Output | Timestamped run directories containing a manifest, compressed NumPy data, and Matplotlib PNG figures |

## Physical picture

### Closed random Hamiltonians

One realization is one random draw of a Hamiltonian $`H`$. Diagonalizing it
gives an eigenspectrum,

```math
H\lvert n\rangle = E_n\lvert n\rangle.
```

Random matrix theory asks for statistical properties shared by many such
draws. The random couplings or matrix-entry distribution fix a global energy
scale. Symmetries fix the allowed matrix structure and the correlations among
its eigenvalues.
The latter are organized by the Dyson index $`\beta`$: $`\beta=1`$ for GOE,
$`\beta=2`$ for GUE, and $`\beta=4`$ for GSE. Poisson spectra provide the
uncorrelated reference case with $`\beta=0`$.

Universality means that local spectral fluctuations are governed mainly by
the symmetry class and remain insensitive to many microscopic details. Level
correlations describe how the location of one energy level changes the likely
locations of the others. Their short-range behavior distinguishes correlated
Wigner–Dyson spectra from independent Poisson levels.

RMTPy uses a many-body parity-sector convention for every concrete ensemble.
With $`N_m`$ Majorana modes, the matrix dimension is

```math
D=2^{N_m/2-1}.
```

The Gaussian and Bogoliubov–de Gennes ensembles supply symmetry-controlled
random matrices directly. The SYK ensemble constructs a Hamiltonian from
random even-$`q`$ products of Majorana operators. The Poisson ensemble draws
independent levels and combines them with a random Wigner–Dyson eigenvector
basis.

Different statistics probe different parts of a spectrum:

- The level density records the broad distribution of energies.
- Nearest-neighbor spacings are adjacent differences in the sorted spectrum
  and measure short-range level repulsion. RMTPy accounts for the ensemble's
  declared eigenvalue degeneracy when forming them.
- A spectral form factor measures correlations in the time domain.

For realization $`r`$, RMTPy accumulates

```math
Z_r(t)=\frac{1}{N}\sum_{j=1}^{N}e^{-iE_{rj}t},
```

and forms

```math
K(t)=\frac{1}{R}\sum_{r=1}^{R}\lvert Z_r(t)\rvert^2,
\qquad
K_{\mathrm{conn}}(t)=\frac{R}{R-1}\left[
K(t)-\left\lvert\frac{1}{R}\sum_{r=1}^{R}Z_r(t)\right\rvert^2
\right],\qquad R>1.
```

Here $`R`$ is the number of realizations and $`N`$ is the number of levels in
the current sample. The connected estimator is zero for a single realization.

Spectral and resonance form-factor archives also retain the first realization's
$`\lvert Z_1(t)\rvert^2`$. Standalone form-factor plots show this trace as
$`K^{(1)}(t)`$ when available, alongside the ensemble averages.

### SYK Hamiltonians and fermions

SYK couples $`N_m`$ Majorana fermions through random interactions among every
set of $`q`$ distinct modes. For the $`q=4`$ model used here, RMTPy's convention
is

```math
H=\sum_{1\le i<j<k<l\le N_m}J_{ijkl}\chi_i\chi_j\chi_k\chi_l,
\qquad \{\chi_i,\chi_j\}=2\delta_{ij}.
```

The couplings are independent real Gaussian variables with

```math
\mathbb{E}[J_{ijkl}]=0,
\qquad \mathbb{E}[J_{ijkl}^2]=\frac{3!J^2}{N_m^3},
```

where $`J`$ is `interaction_strength`. Other supported even-$`q`$ models use
variance $`(q-1)!J^2/N_m^{q-1}`$; the matrix builder includes an imaginary
prefactor when `q % 4 == 2` to keep the Hamiltonian Hermitian. Even-$`q`$
interactions preserve fermion parity, so `is_even_parity` selects one block of
dimension $`D=2^{N_m/2-1}`$. The SYK model and its symmetry class are distinct:
the $`q=4`$, $`N_m=24`$ results below have GOE symmetry, while other supported
configurations can have GUE, GSE, or Poisson references.

[rmtpy/fermions.py](rmtpy/fermions.py) supplies `MajoranaFermionBasis`, which
lazily builds and caches sparse Majorana operators, complex-fermion creation
and annihilation operators, the vacuum, charge conjugation, and a parity slice.
The operators act on the full $`2^{N_m/2}`$-dimensional Fock space. SYK uses
decomposed $`q`$-body monomials restricted to the selected parity block to
assemble each dense Hamiltonian. `SYKCompoundEnsemble` uses the same basis to
create few-fermion channel states, symmetry-compatible couplings, and the
sparse width matrix that opens the system.

### Opening the system

An open system can exchange probability with external channels. RMTPy
represents the channel amplitudes by a coupling matrix $`W`$ and uses the
effective Hamiltonian

```math
H_{\mathrm{eff}}=H-\frac{i}{2}WW^\dagger.
```

Its complex eigenvalues are scattering poles,

```math
\mathcal E_n=E_n-\frac{i}{2}\Gamma_n.
```

The real part $`E_n`$ is a resonance center. The positive quantity
$`\Gamma_n=-2\mathrm{Im}\,\mathcal E_n`$ is its total decay width.
Resolving the coupling in the eigenbasis of the closed Hamiltonian gives the
partial width of each state into each channel. In units with $`\hbar=1`$, width
sets an inverse lifetime, so a narrow resonance is long lived.

At a real probe energy $`E`$, the code evaluates the reaction and scattering
matrices with the conventions

```math
K(E)=\frac{1}{2}W^\dagger(E-H)^{-1}W,
\qquad
S(E)=[I+iK(E)]^{-1}[I-iK(E)].
```

The Wigner–Smith matrix

```math
Q(E)=-iS^\dagger(E)\frac{\mathrm dS(E)}{\mathrm dE}
```

describes the energy sensitivity of the scattering response. Its eigenvalues
are the proper delay times. Channel transmission is calculated from the
ensemble-averaged diagonal scattering amplitude,

```math
T_a(E)=1-\left\lvert\left\langle S_{aa}(E)\right\rangle\right\rvert^2.
```

$`T_a=0`$ describes a closed or fully reflected channel, while $`T_a=1`$
describes ideal transmission.

## Representative results

The following figures come from saved runs of the even-parity SYK model with
$`q=4`$, $`N_m=24`$, $`J=1`$, and 100 realizations.

![Weight-unfolded even-parity SYK nearest-neighbor spacing histogram with GOE, GUE, GSE, and Poisson references](assets/readme/syk-level-spacings.png)

*Weight-unfolded nearest-neighbor spacings from 100 even-parity SYK
realizations with $`q=4`$, $`N_m=24`$, and $`J=1`$. The histogram is compared
with GOE, GUE, and GSE Wigner surmises and the Poisson reference. GOE is the
symmetry class of this SYK configuration.*

![Weight-unfolded two-dimensional histogram of even-parity SYK scattering poles with the energy-resolved average width](assets/readme/syk-resonance-poles.png)

*Weight-unfolded complex-energy histogram from 100 even-parity SYK compound
realizations with $`q=4`$, $`N_m=24`$, $`J=1`$, $`N_f=2`$, and equal couplings
$`v_a=\sqrt{E_0}\approx2.14384`$, giving $`\binom{12}{2}=66`$ open channels.
Here $`E_0`$ is the SYK spectral radius. The horizontal coordinate remains
$`E/E_0`$; the vertical coordinate is $`\log_{10}\gamma`$, where $`\gamma`$
is the width unfolded with the SYK $`q`$-Hermite weight. The cyan curve is the
energy-resolved average unfolded width.*

![Weight-unfolded even-parity SYK proper-delay histogram at energy zero with matching weight-unfolded spectral form factors overlaid](assets/readme/syk-time-delay-sff-overlay.png)

*Weight-unfolded proper-delay distribution at $`E=0`$ from 100 even-parity SYK
compound realizations with $`q=4`$, $`N_m=24`$, $`J=1`$, $`N_f=2`$, and
$`v_a=\sqrt{E_0}\approx2.14384`$ for all 66 channels. The yellow histogram and
black Brouwer–Frahm–Beenakker (BFB) reference use the linear left axis. The
matching weight-unfolded spectral run supplies the blue $`K(\upsilon)`$ and
orange $`K_{\mathrm{conn}}(\upsilon)`$ on the logarithmic right axis; the
dotted curve is the universal GOE connected form factor for this SYK symmetry
class.*

![Energy-dependent channel-zero transmission coefficient for an even-parity SYK compound](assets/readme/syk-transmission-coefficient.png)

*Channel-zero transmission from 100 even-parity SYK compound realizations
with $`q=4`$, $`N_m=24`$, $`J=1`$, $`N_f=2`$, and
$`v_a=\sqrt{E_0}\approx2.14384`$ for all 66 channels. The 500-point grid spans
1.5 times the density plotting interval. The calculation averages $`S_{00}(E)`$
before forming $`T_0(E)=1-\lvert\langle S_{00}(E)\rangle\rvert^2`$. The
horizontal coordinate is $`E/E_0`$; transmission has no unfolding variant.*

## Installation

RMTPy currently runs from a source checkout. The project configuration requires
Python 3.14. Create an environment with the runtime dependencies and Ruff, then
run Python from the repository root:

```bash
git clone https://github.com/joshua-leeman/RMTPy.git
cd RMTPy

conda create -n rmtpy-env -c conda-forge \
  python=3.14 numpy scipy numba attrs cattrs matplotlib ruff
conda activate rmtpy-env
```

NumPy supplies arrays and random generators. SciPy supplies sparse matrices,
interpolation, special functions, and BLAS/LAPACK access. Numba compiles the
matrix and polynomial kernels. `attrs` and `cattrs` provide validated records
and structured conversion. Matplotlib produces the figures.

Numerical calculations use no TeX subprocess. Plot generation configures
Matplotlib to use LaTeX, `amsmath`, and the Latin Modern Roman font. Install a
TeX distribution containing those components before calling the plotting
methods or the `run_*_simulation` helpers.

Public imports begin at `rmtpy.ensembles`, `rmtpy.compounds`, and
`rmtpy.simulations`. The top-level `rmtpy` package defines no convenience
imports.

```python
from rmtpy.compounds import SYKCompoundEnsemble
from rmtpy.ensembles import SYK
from rmtpy.simulations import SpectralStatisticsSimulation
```

## Quick start

### Stream one closed and one open realization

This example draws an even-parity $`q=4`$ SYK spectrum, opens the same ensemble
with six channels, and evaluates its poles and delay times.

```python
import numpy as np

from rmtpy.compounds import SYKCompoundEnsemble
from rmtpy.ensembles import SYK

# N_m = 8 gives one parity block with dimension D = 8.
ensemble = SYK(
    q=4,
    num_majoranas=8,
    is_even_parity=True,
    max_spectral_polynomial_degree=0,
    seed=2025,
)

levels = next(ensemble.eigvals_stream(realizs=1)).copy()
spacings = np.diff(levels)

print(ensemble.dimension)           # 8
print(ensemble.universality_class)  # GOE: the symmetry class of this SYK model
print(levels.shape)                 # (8,)
print(spacings.shape)               # (7,)

# Even N_f matches even parity; N_f = 2 gives C = binom(4, 2) = 6 channels.
compound = SYKCompoundEnsemble(
    ensemble=ensemble,
    num_free_complex_fermions=2,
)

poles = next(compound.resonances_stream(realizs=1)).copy()
resonance_centers = poles.real
resonance_widths = -2.0 * poles.imag

delay_times, closed_levels = next(
    compound.time_delays_stream(
        realizs=1,
        energies=np.array([0.0]),
    )
)
delay_times = delay_times.copy()
closed_levels = closed_levels.copy()

print(compound.num_channels)   # 6
print(poles.shape)             # (8,)
print(delay_times.shape)       # (1, 6): one energy, six channels
```

Methods ending in `_stream` yield one realization at a time. Several streams
reuse their working arrays for subsequent draws. Copy a yielded array when it
must remain unchanged after the iterator advances.

### Inspect the fermionic basis

The SYK `ensemble` above owns a `MajoranaFermionBasis`. Access its sparse
operators and vacuum directly:

```python
basis = ensemble.majorana_fermion_basis
majoranas = basis.majorana_fermions
annihilation, creation = basis.complex_fermions
vacuum = basis.vacuum_state
parity_slice = basis.parity_block_slice

print(basis.num_complex_fermions)  # 4
print(basis.dimension)            # 16: the full Fock space
print(majoranas[0].shape)          # (16, 16)
print(vacuum.shape)               # (16, 1)
print(ensemble.dimension)         # 8: the selected parity block
```

The basis reuses these cached objects when building the SYK interactions and
the compound's few-fermion channel states.

### Run, save, plot, and reload a simulation

A simulation combines many realizations into finalized histograms and moment
averages. The following example uses a small degree-zero configuration. It
constructs raw and weight-unfolded statistics without polynomial calibration.

```python
from pathlib import Path

from rmtpy.ensembles import SYK
from rmtpy.simulations import (
    SpectralStatisticsSimulation,
    load_spectral_statistics_simulation,
)

simulation = SpectralStatisticsSimulation(
    ensemble=SYK(
        q=4,
        num_majoranas=8,
        is_even_parity=True,
        max_spectral_polynomial_degree=0,
        seed=7,
    ),
    realizs=64,
)

simulation.execute()

# Finalized Data objects remain on the completed simulation.
raw_levels = simulation.raw_buffers.levels
print(raw_levels.bins)
print(raw_levels.histogram)

run_dir = simulation.save(Path("outputs"))
simulation.plot(run_dir)  # Requires the LaTeX setup described above.

restored = load_spectral_statistics_simulation(directory=run_dir)
print(restored.raw_buffers.levels.histogram)
```

`execute()` fills the simulation in place and returns `None`. `save()` returns
the exact timestamped run directory. `plot()` reads the saved data in that
directory and writes each PNG beside its NPZ file. The family loader rebuilds a
completed simulation with its data and final random-generator state.

Every family also exports a `run_*_simulation` helper. For example,
`run_spectral_statistics_simulation(...)` constructs the simulation, executes
it, saves it under `outputs/`, reloads the saved run for plotting, and returns
the completed in-memory simulation.

## Ensemble catalogue

The concise aliases in the first column are exported alongside the full class
names.

| Alias | Class | Matrix or spectral structure | $`\beta`$ | Density weight and polynomial basis |
| --- | --- | --- | ---: | --- |
| `GOE` | `GaussianOrthogonalEnsemble` | Real symmetric Gaussian matrix | 1 | Semicircle and Chebyshev-$`U`$ |
| `GUE` | `GaussianUnitaryEnsemble` | Complex Hermitian Gaussian matrix | 2 | Semicircle and Chebyshev-$`U`$ |
| `GSE` | `GaussianSymplecticEnsemble` | Self-dual Hermitian matrix with Kramers degeneracy | 4 | Semicircle and Chebyshev-$`U`$ |
| `BdGC` | `BogoliubovDeGennesCEnsemble` | Particle-hole-symmetric block matrix | 2 | Semicircle and Chebyshev-$`U`$ |
| `BdGD` | `BogoliubovDeGennesDEnsemble` | Imaginary antisymmetric Hermitian matrix | 2 | Semicircle and Chebyshev-$`U`$ |
| `Poisson` | `PoissonEnsemble` | Independent uniform levels with a selectable Wigner–Dyson eigenvector basis | 0 | Uniform and Legendre |
| `SYK` | `SachdevYeKitaevEnsemble` | Random even-$`q`$ Majorana Hamiltonian | Derived from $`q`$ and $`N_m`$ | $`q`$-Hermite |

All concrete constructors are keyword-only. They share these controls:

- `num_majoranas` is even and lies between 4 and 32. It sets
  $`D=2^{N_m/2-1}`$.
- `interaction_strength` is positive and defaults to 1. For the Gaussian,
  BdG, and Poisson ensembles, the nominal spectral radius is $`E_0=N_mJ`$.
- `max_spectral_polynomial_degree` is a nonnegative integer and defaults to 6.
- `dtype` defaults to `complex128`. The ensemble derives the corresponding
  real and complex working dtypes.
- `seed` accepts the forms supported by `numpy.random.default_rng`.

At the lowest level, `matrix_stream()`, `eigsys_stream()`, and
`eigvals_stream()` produce matrices, eigensystems, and eigenvalues one
realization at a time.

`PoissonEnsemble.eigvec_ensemble_flag` accepts `"GOE"`, `"GUE"`, or `"GSE"`
and defaults to `"GUE"`. The level and eigenvector generators share one NumPy
random generator, so their sampling order is reproducible from the same initial
state.

SYK accepts `q` in `{2, 4, 6, 8, 10}` with `q < num_majoranas` and an
`is_even_parity` flag. Its supported size limits reflect storage for the
decomposed Majorana monomials:

| $`q`$ | Maximum $`N_m`$ |
| ---: | ---: |
| 2 or 4 | 32 |
| 6 | 26 |
| 8 | 24 |
| 10 | 22 |

For `q=2`, the code uses Poisson spectral references. For larger `q`, it uses
GOE when `q % 4 == 0` and `num_majoranas % 8 == 0`, GSE when `q % 4 == 0` and
`num_majoranas % 8 == 4`, and GUE for the remaining cases.

The `MajoranaFermionBasis` builds its expensive sparse objects lazily. These
include the Majorana operators, creation and annihilation operators, the
vacuum, charge conjugation, the selected parity slice, and decomposed
$`q`$-body monomials.

## Compound ensembles and open channels

Three public compound classes cover the open-system constructions:

| Class | Closed source and coupling construction |
| --- | --- |
| `CompoundEnsemble` | A non-Poisson many-body ensemble with basis-state channel couplings |
| `PoissonCompoundEnsemble` | A Poisson spectrum with channel rotation supplied by its selected eigenvector ensemble |
| `SYKCompoundEnsemble` | An SYK model with few-fermion channel states and a sparse width matrix |

If `num_free_complex_fermions` is $`N_f`$, the number of open channels is

```math
C=\binom{N_m/2}{N_f}.
```

The integer $`N_f`$ ranges from zero through $`N_m/2`$ and defaults to one. The
channel count must fit within the Hilbert-space dimension. Use
`PoissonCompoundEnsemble` for a `PoissonEnsemble`; `CompoundEnsemble` accepts
the other many-body ensembles. The `couplings` argument accepts one positive
finite scalar or a real, finite, nonnegative sequence of length $`C`$. RMTPy
copies a supplied sequence and marks the stored array read-only. The default
coupling for every channel is the square root of the ensemble spectral radius.

`SYKCompoundEnsemble` requires the parity of $`N_f`$ to match the selected SYK
parity sector. A symplectic SYK compound also requires equal coupling strengths
within each Kramers pair.

Compound objects expose the following streams:

- `effective_hamiltonian_stream()` and `resonances_stream()`;
- `resonance_real_parts_stream()` and `partial_widths_stream()`;
- `reaction_matrix_stream()`, `reaction_matrix_pair_stream()`, and
  `scattering_matrix_stream()` over an energy array;
- `wigner_smith_matrix_stream()` and `time_delays_stream()`.

## Simulations

RMTPy provides five completed Monte Carlo simulation families. Each class is
available from `rmtpy.simulations` and from its family subpackage.

| Simulation | Required scientific inputs | Finalized products |
| --- | --- | --- |
| `SpectralStatisticsSimulation` | `ensemble`, `realizs` | Density-coefficient histograms, levels, nearest-neighbor spacings, spectral form factors, and connected form factors |
| `ResonanceStatisticsSimulation` | `compound`, `realizs` | Resonance-density coefficients, centers, widths, spacings, two-dimensional complex-energy densities, and resonance form factors |
| `PartialWidthsStatisticsSimulation` | `compound`, `width_indices`, `realizs` | Selected partial-width and total-width histograms, each scaled by its observed mean |
| `TimeDelayStatisticsSimulation` | `compound`, `energies`, `realizs` | Proper-delay histograms for every requested probe energy, with optional matching spectral-form-factor overlays |
| `TransmissionCoefficientsSimulation` | `compound`, `channel_indices`, `realizs` | Energy-resolved transmission for selected channels and one all-channel Weisskopf estimate |

Partial-width selectors have two forms. `(state, channel)` selects one partial
width. `(state,)` selects the total width of that state, summed over all
channels. Selectors must be unique and within the state and channel bounds.

Time-delay probe energies are copied into a contiguous, read-only `float64`
array. The input must be scalar or one-dimensional, finite, nonempty, and free
of duplicate values. Pass a completed spectral run as
`spectral_statistics_directory` when plotting or using the run helper to
overlay its matching raw and unfolded form factors. The ensemble configuration
and time range must match; its seed and realization count may differ.

Transmission simulations use a fixed 500-point grid spanning 1.5 times the
ensemble density's plotting range. `channel_indices` chooses the individual
$`T_a(E)`$ curves retained for output. The accompanying Weisskopf estimate

```math
\Gamma_{\mathrm W}(E)=\frac{d(E)}{2}\sum_{a=1}^{C}T_a(E),
\qquad
d(E)=\frac{1}{D\rho_{\mathrm{weight}}(E)},
```

always includes every channel in the compound. Points with zero spectral
weight are stored as undefined values.

### Execution lifecycle

Each simulation instance accepts one execution attempt:

```text
NEW ──► RUNNING ──► COMPLETE
              └──► FAILED
```

Completion and failure are terminal states. A completed simulation is
iterable; iteration yields each finalized `Data` object in its defined storage
order. Saving and plotting require `COMPLETE`. Plotting also requires the NPZ
files already written by `save()`.

Every family exports the same function pattern:

- `run_<family>_simulation(...)` executes, saves, plots, and returns the
  completed simulation;
- `load_<family>_simulation(directory=...)` reconstructs a saved simulation;
- `plot_<family>_simulation(directory=...)` loads a run and dispatches every
  stored data object to its plot class.

## Density estimation and unfolding

Raw energies contain the smooth variation of the mean density together with
the fluctuations under study. Unfolding maps the energy coordinate through a
cumulative distribution function $`F`$:

```math
\widetilde E=D[F(E)-F(0)].
```

One unit in the unfolded coordinate corresponds approximately to one local
mean level spacing. A finite resonance width is transformed over its full
energy interval:

```math
\widetilde\Gamma=
D\left[F\left(E+\frac{\Gamma}{2}\right)-
F\left(E-\frac{\Gamma}{2}\right)\right].
```

Spectral, resonance, and time-delay simulations construct four views where the
underlying density supports them:

| View | CDF used by the calculation |
| --- | --- |
| Raw | Original energies or widths in model units |
| Weight-unfolded | The ensemble's leading semicircle, uniform, or $`q`$-Hermite weight |
| Average-unfolded | Polynomial coefficients averaged over additional ensemble samples, with one result for every truncation degree |
| Variate-unfolded | Coefficients fitted to each realization, with one result for every truncation degree |

For a density histogram with bin count $`n_i`$ and bin width $`\Delta_i`$, the
finalized value is

```math
h_i=\frac{n_i}{\left(\sum_j n_j\right)\Delta_i}.
```

This normalization integrates to one over the samples that fall inside the
declared histogram support.

For `max_spectral_polynomial_degree=M`, coefficient histograms and both average
and variate unfolding groups are created for every degree from 1 through
$`M`$, including odd degrees. Setting `M=0` leaves the raw and weight-unfolded
groups and avoids the polynomial groups and averaged-density calibration. For
densities with a polynomial expansion, the degree-zero raw spectral and
resonance plots still overlay the polynomial weight as a reference curve.

Spectral and resonance coefficient samples must be finite. A coefficient of
degree $`n`$ uses a deterministic symmetric support
$`\pm(n+1)/\sqrt{D}`$, where $`D`$ is the density dimension, and
$`\max(100,\lceil\sqrt{D}\rceil)`$ bins. Runs with the same scientific
configuration therefore have identical grids regardless of seed or realization
count. Samples outside the finite grid are recorded in explicit underflow and
overflow counters, and archives identify the rule with
`metadata["grid_policy"] == "configuration_v1"`. The plots use a symmetric
window around the central 1st–99th-percentile in-range mass, and all coefficient
plots in one spectral or resonance run share the widest such horizontal scale.

Spectral statistics use `ensemble.spectral_density`. Resonance statistics use
`compound.resonance_density` and fit the resonance centers. In an unfolded
two-dimensional complex-energy histogram, the horizontal axis remains the
physical center $`E/E_0`$, while the vertical width is unfolded. Time-delay
statistics transform $`1/\tau`$ as a width about the fixed probe energy and
then take the reciprocal.

An averaged density calibration draws

```text
max(8192 // dimension, 10)
```

additional samples when its coefficient cache is first needed. Calibration
and the main simulation use the same ensemble random generator. The initial
cache state therefore forms part of the sampling order, and the manifest
records whether calibration was already cached or was computed during
execution.

## Analytical references

`rmtpy.universal` contains the formulas used by the plots and direct ensemble
methods:

- symmetry labels and eigenvalue degeneracies for $`\beta=0,1,2,4`$;
- Poisson and Wigner-surmise spacing densities;
- single-channel and multi-channel Porter–Thomas width densities;
- connected GOE, GUE, GSE, and Poisson spectral form factors;
- the compact-support proper-delay density used for the time-delay comparison.

The `DensityModel` in `rmtpy.density` combines a support, a sample stream, an
optional orthogonal-polynomial family, and its weight. It supplies PDFs, CDFs,
ensemble-average coefficients, realization-specific coefficients, and the
interpolators used by unfolding. `rmtpy.polynomials` implements the
Chebyshev-$`U`$, Legendre, and $`q`$-Hermite recurrences and weights.
`rmtpy.fermions` supplies the sparse Majorana and complex-fermion operators,
parity projections, $`q`$-body monomials, and few-fermion channel states used by
the SYK constructions.

## Slurm jobs

The top-level `slurm` package validates a simulation locally and writes a
portable Slurm batch script. Its built-in configuration is deliberately small
and cluster-neutral: one node, one task, one CPU per task, a one-hour limit,
cyclic distribution, and no assumed partition, modules, conda environment,
mail address, CPU binding, or NUMA layout.

Run the generator from the RMTPy repository root and state the resources needed
by the first test job explicitly:

```bash
python -m slurm \
  --simulation spectral-statistics \
  --ensemble "{'name': 'SYK', 'q': 4, 'N': 16, 'seed': 123}" \
  --realizs-per-task 2 \
  --directory outputs/slurm_tutorial \
  --job-name syk_smoke \
  --output syk_smoke.slurm \
  --nodes 1 \
  --tasks-per-node 4 \
  --cpus-per-task 2 \
  --time 01:00:00 \
  --no-plot

bash -n syk_smoke.slurm
sbatch syk_smoke.slurm
```

### Private cluster profiles

Site-specific queue names, limits, modules, environments, and memory policies
belong in a private TOML profile rather than public Python. Copy
[`slurm/profile.example.toml`](slurm/profile.example.toml) to the ignored
`.rmtpy-slurm.toml` path or to a location outside the repository, replace its
fictional values with the cluster's documented values, and select it through an
environment variable:

```bash
export RMTPY_SLURM_PROFILE_FILE=.rmtpy-slurm.toml
python -m slurm --help
```

An existing `.env` file can hold that export and other private values, but the
package does not silently load `.env` or require `python-dotenv`. From the
repository root, import its values into the current shell with:

```bash
set -a
source .env
set +a
```

Explicit command-line values override profile values; profile values override
the generic defaults. A profile can define partition-specific core counts,
maximum nodes, maximum wall time, required memory, and a NUMA default. With no
partition table, `--partition` accepts any site-defined name and leaves final
validation to Slurm.

The generator supports `--nodes`, `--tasks-per-node`, `--cpus-per-task`,
`--time`, `--partition`, `--memory`, and `--distribution`. Environment setup is
optional: use repeated `--module` flags, `--module-purge`/`--no-module-purge`,
`--conda-env`/`--no-conda`, `--cpu-bind`/`--no-cpu-bind`, and
`--numactl`/`--no-numactl` with `--numa-policy`. `--no-modules` and
`--no-partition` clear corresponding profile defaults.

### Private notification addresses

Generated scripts never contain `#SBATCH --mail-user`. Slurm reads `#SBATCH`
directives before the shell runs and therefore does not expand an environment
variable placed in a directive. If the cluster's submitting-user default is
not sufficient, keep the address in a private environment variable and pass it
only while submitting:

```bash
export RMTPY_SLURM_MAIL_USER=user@example.com
sbatch --mail-user="$RMTPY_SLURM_MAIL_USER" \
  --mail-type=BEGIN,END,FAIL syk_smoke.slurm
```

See the official [`sbatch` documentation](https://slurm.schedmd.com/sbatch.html)
for directive and command-line precedence. Generated `*.slurm` files are
ignored because explicitly supplied paths or other local settings may still be
private.

### Scientific inputs and distributed execution

The mapping options accept JSON or safe Python-literal syntax. Common compact
aliases are normalized: `N` means `num_majoranas`, `J` means
`interaction_strength`, `max_degree` means
`max_spectral_polynomial_degree`, and `parity` means `is_even_parity`.
Compound mappings similarly accept `Nf` for `num_free_complex_fermions` and
`v` for `couplings`. The appropriate SYK, Poisson, or generic compound class is
inferred when the compound `name` is omitted.

Each family uses the following `--simulation-args` mapping:

| `--simulation` | Simulation arguments |
| --- | --- |
| `spectral-statistics` | `{}` (the default) |
| `resonance-statistics` | `{}` (the default) |
| `partial-widths-statistics` | Required `width_indices`, for example `{'width_indices': [[0, 0], [0]]}` |
| `time-delay-statistics` | Required `energies`; optional `spectral_statistics_directory` for the plot overlay |
| `transmission-coefficients` | Required `channel_indices`, for example `{'channel_indices': [0, 1]}` |

Generation constructs the actual ensemble, compound, and simulation objects,
so invalid scientific combinations fail before allocation. If no seed is
given, the generator materializes one and embeds it in the normalized
specification. Calibration and production workers receive separate,
reproducible NumPy `SeedSequence` children.

`--realizs-per-task` is a per-task count:

```text
total realizations = nodes × tasks per node × realizations per task
```

Workers write partial RMTPy archives to the cluster's shared filesystem. One
coordinator pools density calibration, merges additive accumulators, finalizes
one ordinary RMTPy simulation, records distributed provenance in its manifest,
and plots once. Temporary state lives under
`<directory>/.rmtpy-slurm/$SLURM_JOB_ID`; it is removed after successful
publication and retained after failure. The generator creates the log directory
and refuses to replace an existing script unless `--force` is supplied. Use
`--no-plot` when the compute environment lacks RMTPy's TeX plotting tools.

## Saved runs and plots

`Simulation.save(root)` writes beneath a readable path derived from the
simulation family and physical source. Selected entries from a spectral run
have the form

```text
outputs/
└── spectral_statistics_simulation/
    └── SYK_4_even/Nm_24/J_1p0/max_polydeg_2/realizs_100/
        └── 2026-10-04T04:19:02.900442Z/
            ├── manifest.json
            ├── spectral_histogram/
            │   ├── spectral_histogram_data.npz
            │   └── spectral_histogram_plot.png
            ├── spacings_histogram/
            │   ├── spacings_histogram_data.npz
            │   └── spacings_histogram_plot.png
            └── spectral_form_factors/
                ├── spectral_form_factors_data.npz
                └── spectral_form_factors_plot.png
```

Compound paths add $`N_f`$ and, for equal couplings, an `alpha_...` label based
on $`\alpha=\log_{10}(\langle v_a^2\rangle/E_0)`$. Nonuniform couplings use a
stable `couplingsID_...` identifier instead. Each execution receives a UTC
timestamp directory.

The manifest records:

- the simulation class and normalized constructor configuration;
- the seed policy, bit generator, and initial and final RNG states;
- the real and complex NumPy dtypes;
- execution state and UTC completion time;
- averaged-density calibration coefficients and their timing when applicable.

Each data object is stored as a compressed NPZ archive with its concrete Python
class, initializer fields, numeric arrays, and metadata. Loading uses
`allow_pickle=False`. Plot classes receive the saved manifest, reconstruct the
required ensemble or compound as detached plotting context, and use the
recorded calibration. A PNG is written atomically beside its data archive, and
an existing plot path is preserved.

### Aggregating cluster jobs

`Simulation.aggregate(superfolder)` combines completed, embarrassingly parallel
runs without modifying their archives. The superfolder must contain immediate
children named `job_<integer>_outputs`; each child must contain exactly one saved
run of the requested concrete simulation class. Seeds and per-job realization
counts may differ, while every scientific parameter and dtype must match.

For example, collect saved SYK spectral runs with matching $`q`$, $`N_m`$,
parity, interaction strength, and polynomial degree under `syk_cluster_results`:

```python
from rmtpy.simulations import SpectralStatisticsSimulation

aggregate = SpectralStatisticsSimulation.aggregate("syk_cluster_results")
aggregate_directory = aggregate.save("syk_aggregated_outputs")
aggregate.plot(aggregate_directory)
```

The returned simulation is already complete. Its realization count is the sum
of the workers, its derived statistics are recomputed from summed counts and
moments, and its manifest records the ordered source paths, seeds, realization
counts, completion times, and pooled density calibration. It also retains each
source calibration and records `unfolding_policy="pooled_source_calibrations"`.
Histograms unfolded with different source calibrations are pooled as saved;
the weighted calibration summarizes those sources. Re-unfolding all samples
with one global calibration would require the original spectra.
The source fields live
under `execution["aggregation"]`; the RNG policy is `"aggregate"`, with ordered
source seeds in `rng["seed"]`. Calling the base `Simulation.aggregate` is
supported only when each job directory contains one unambiguous simulation
class.

Archives written before deterministic coefficient grids remain loadable and
plottable, but they cannot be aggregated exactly because their adaptive bin
edges may differ. Aggregation rejects those coefficient archives with an
explicit error. Legacy partial-width archives remain exactly aggregatable: the
unnormalized width sum is recovered from their saved mean and realization
count.

Legacy `averaged` unfolding names are accepted alongside `average`. Saved
transmission energy grids are restored from their archives, including older
100-point grids. If an older form-factor archive lacks its first-realization
trace, that trace is recovered for a single-realization run. For larger runs,
the trace is omitted from plots because aggregate moments cannot recover it.

## Numerical behavior and limits

- Matrix construction and polynomial recurrences use Numba. The first call to
  a compiled kernel includes JIT compilation time.
- Eigensystems use SciPy BLAS/LAPACK. Dense diagonalization costs
  $`O(D^3)`$, while $`D=2^{N_m/2-1}`$ grows exponentially.
- SYK stores decomposed data for $`\binom{N_m}{q}`$ Majorana monomials. Its
  constructor limits guard broad memory bounds; a large accepted case can
  still exceed the practical resources of a particular machine.
- Fermion operators and SYK channel matrices use SciPy sparse arrays. The
  realized Hamiltonians remain dense for diagonalization.
- Ensemble and compound streams retain one realization at a time. Histogram
  counts and form-factor moments accumulate in fixed-size buffers.
- Histograms use fixed half-open supports. Samples outside ordinary histogram
  supports are omitted; coefficient histograms additionally retain explicit
  underflow and overflow counts so parallel runs remain exactly additive.
- Seeded runs reproduce the NumPy random trajectory within the same numerical
  software stack. BLAS/LAPACK, NumPy, SciPy, and Numba versions can affect
  bitwise results.

Begin with small `num_majoranas`, a modest realization count, and polynomial
degree zero. Increase one cost driver at a time after checking memory use and
the output supports.

## Repository layout

```text
rmtpy/
├── ensembles/       # closed Gaussian, BdG, Poisson, and SYK sources
├── compounds/       # open-channel coupling and scattering quantities
├── simulations/     # Monte Carlo accumulation, persistence, and plotting
├── density.py       # density models, PDFs, CDFs, and coefficient caches
├── fermions.py      # sparse Majorana and complex-fermion algebra
├── polynomials.py   # Chebyshev, Legendre, and q-Hermite systems
├── universal.py     # analytical universal reference laws
├── conversion.py    # structured conversion plus path and LaTeX labels
└── validators.py    # shared numerical validation

slurm/               # cluster-neutral Slurm generation and distributed execution
tests/               # numerical, lifecycle, persistence, plotting, and API tests
assets/readme/       # figures embedded in this README
pyproject.toml       # Python, Ruff/Black, and Cursor recommended typing settings
```

The code uses Python 3.14 type syntax, keyword-only frozen `attrs` records for
scientific state, dataclasses for mutable plot configuration, and explicit
validators at array and serialization boundaries.

## Tests and formatting

Run the verification suite from the repository root with the environment
active:

```bash
python -m unittest discover -s tests -v
ruff check rmtpy tests slurm notebooks
ruff format --check rmtpy tests slurm notebooks
```

The tests cover analytical and matrix-level numerical regressions, ensemble and
compound behavior, histogram normalization, unfolding, simulation lifecycle,
all five simulation families, save/load restoration, plotting, and the public
simulation exports.

Cursor Pyright uses recommended mode with the Python 3.14 scientific interpreter.
The project settings cover the library, tests, and Slurm tools; local notebook
code is checked separately. Errors and warnings should both be resolved.

## License

RMTPy is available under the [MIT License](LICENSE). Copyright © 2025 Joshua
Leeman.
