# rmtpy.simulations.cdo_evolution.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
import itertools
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from math import comb
from multiprocessing import Pool, set_start_method, shared_memory
from time import time
from typing import Any

# Third-party imports
import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.special import jn_zeros

# Local application imports
from rmtpy.plotting.cdo_evolution.probabilities_plot import ProbabilitiesPlot
from rmtpy.plotting.cdo_evolution.purity_plot import PurityPlot
from rmtpy.plotting.cdo_evolution.entropy_plot import EntropyPlot
from rmtpy.plotting.cdo_evolution.observable_plot import ObservablePlot
from rmtpy.simulations._mc import MonteCarlo, _parse_mc_args
from rmtpy.special import create_majorana_pairs
from rmtpy.utils import get_ensemble, configure_matplotlib


# =======================================
# 2. Functions
# =======================================
def calculate_cdo(evolved_states: np.ndarray) -> np.ndarray:
    """Calculate CDO from evolved states."""
    # Unpack number of realizations
    realizs = evolved_states.shape[0]

    # Compute CDOs
    cdo = np.matmul(evolved_states.conj().T, evolved_states)
    cdo /= realizs

    # Return CDOs
    return cdo


def cdo_expectation(states: np.ndarray, observable: np.ndarray) -> np.ndarray:
    """Calculate expectation value of observable from CDO."""
    # Sum over different realized expectation values
    expectation = np.einsum("trd,dk,trk->t", states.conj(), observable, states)

    # Divide by number of realizations
    expectation /= states.shape[1]

    # Return expectation value
    return expectation.real


def cdo_probabilities(cdo: np.ndarray) -> np.ndarray:
    """Calculate probabilities from CDO."""
    # Return diagonal elements of CDO
    return np.diagonal(cdo).real


def cdo_classic_purity(cdo: np.ndarray) -> np.ndarray:
    """Calculate classical purity from CDO."""
    # Compute population purity from CDO and return it
    return np.sum(cdo_probabilities(cdo) ** 2).real


def cdo_nonlinears(cdo: np.ndarray, overwrite: bool = False) -> np.ndarray:
    """Calculate quantum purity and von Neumann entropy from CDO."""
    # Compute eigenvalues of CDO
    eigvals = eigvalsh(cdo, overwrite_a=overwrite, check_finite=False)

    # Clip eigenvalues to avoid numerical issues
    eigvals = eigvals[eigvals > 1e-10]

    # Calculate quantum purity from eigenvalues
    q_purity = np.sum(eigvals**2)

    # Calculate von Neumann entropy from eigenvalues
    entropy = -np.sum(eigvals * np.log(eigvals))

    # Return quantum purity and von Neumann entropy
    return q_purity, entropy


def _calc_evolved_states(args: dict[str, Any]) -> dict[np.ndarray]:
    """Worker function to calculate evolved states."""
    # Unpack ensemble arguments
    ens_args = args["ens_args"]

    # Find and initialize ensemble
    ensemble = get_ensemble(ens_args)

    # Unpack observable q-parameter
    obs_q = args["obs_q"]

    # Unpack unfolding flag
    unfold = args["unfold"]

    # Unpack worker ID
    worker_id = args["worker_id"]

    # Unpack number of workers
    num_workers = args["num_workers"]

    # Unpack number of realizations
    realizs = args["realizs"]

    # Calculate number of realizations per worker and remainder
    realizs_per_worker, remainder = divmod(realizs, num_workers)

    # Initialize realizations array
    realizs_array = np.full(num_workers, realizs_per_worker, dtype=int)

    # Distribute remainder realizations to workers
    realizs_array[:remainder] += 1

    # Initialize configuration
    config = Config(**args["config"])

    # Unpack shared memory name
    shm_name = args["shm_name"]

    # Store first positive zero of 1st Bessel function
    j_1_1 = jn_zeros(1, 1)[0]

    # Create times array based on unfolding flag
    if unfold:
        # Create unfolded times array
        times = config._create_unf_times_array(base=ensemble.dim)
        times *= 2 * np.pi
    else:
        # Create times array
        times = config._create_times_array(base=ensemble.dim)
        times *= j_1_1 / ensemble.E0

    # Find shared memory block
    shm = shared_memory.SharedMemory(name=shm_name)

    # View shared memory block as a np.ndarray
    evolved_states = np.ndarray(
        (config.num_times, realizs, ensemble.dim),
        dtype=ensemble.dtype,
        buffer=shm.buf,
        order="F",
    )

    # Construct initial state
    initial_state = config._construct_initial_state(ensemble.N, ensemble.dtype, obs_q)

    # Determine beginning and ending indices for each worker
    start_idx = np.sum(realizs_array[:worker_id])
    end_idx = start_idx + realizs_array[worker_id]

    # Evolve states
    ensemble.evolve_states(
        state=initial_state,
        times=times,
        realizs=realizs_array[worker_id],
        unfold=unfold,
        out=evolved_states[:, start_idx:end_idx, :],
    )

    # Close shared memory block
    shm.close()


def _calc_cdo_stats(args: dict[str, Any]) -> dict[np.ndarray]:
    """Worker function to perform CDO statistics."""
    # Unpack ensemble arguments
    ens_args = args["ens_args"]

    # Find and initialize ensemble
    ensemble = get_ensemble(ens_args)

    # Unpack unfolding flag
    unfold = args["unfold"]

    # Unpack observable q-parameter
    obs_q = args["obs_q"]

    # Unpack worker ID
    worker_id = args["worker_id"]

    # Unpack number of workers
    num_workers = args["num_workers"]

    # Unpack number of realizations
    realizs = args["realizs"]

    # Initialize configuration
    config = Config(**args["config"])

    # Unpack shared memory name
    shm_name = args["shm_name"]

    # Store first positive zero of 1st Bessel function
    j_1_1 = jn_zeros(1, 1)[0]

    # Create times array based on unfolding flag
    if unfold:
        # Create unfolded times array
        times = config._create_unf_times_array(base=ensemble.dim)
        times *= 2 * np.pi
    else:
        # Create times array
        times = config._create_times_array(base=ensemble.dim)
        times *= j_1_1 / ensemble.E0

    # Array-split times for each worker and delete times
    times_split = np.array_split(times, num_workers)
    del times

    # Grab subset of times for given worker_id
    times = times_split[worker_id]

    # Determine beginning and end indices for given worker_id
    start = int(np.sum([len(t) for t in times_split[:worker_id]]))
    end = start + len(times)

    # Find shared memory block
    shm = shared_memory.SharedMemory(name=shm_name)

    # View shared memory block as a np.ndarray
    states = np.ndarray(
        (config.num_times, realizs, ensemble.dim),
        dtype=ensemble.dtype,
        buffer=shm.buf,
        order="F",
    )

    # Initialize array to store CDO statistics
    cdo_stats = np.empty((len(times), 8), dtype=ensemble.real_dtype)

    # Store times in first column of cdo_stats
    cdo_stats[:, 0] = times

    # Construct observable matrix
    observable = config._construct_observable(ensemble.N, ensemble.dtype, obs_q)

    # Calculate expectation values of observable
    cdo_stats[:, 6] = cdo_expectation(states[start:end, :, :], observable)

    # Calculate second moments of observable
    cdo_stats[:, 7] = cdo_expectation(states[start:end, :, :], observable @ observable)

    # Delete observable to free memory
    del observable

    # Loop over times, calculate CDO, and perform statistics
    for i, t in enumerate(range(start, end)):
        # Calculate CDO
        cdo = calculate_cdo(states[t, :, :])

        # Calculate probabilities
        cdo_stats[i, 1:3] = cdo_probabilities(cdo)[:2]

        # Calculate classical purity and remove bias
        cdo_stats[i, 4] = cdo_classic_purity(cdo)

        # Calculate quantum purity and von Neumann entropy
        q_purity, entropy = cdo_nonlinears(cdo, overwrite=True)

        # Store bias-corrected quantum purity in cdo_stats
        cdo_stats[i, 3] = q_purity

        # Store bias-corrected entropy in cdo_stats
        cdo_stats[i, 5] = entropy

        # Delete CDO to free memory
        del cdo

    # Close shared memory block
    shm.close()

    # Return CDO statistics
    return cdo_stats


def _parse_cdo_mc_args(parser: ArgumentParser) -> dict[str, Any]:
    """Parse command line arguments for CDO Monte Carlo simulation."""
    # Add unfolding flag to parser
    parser.add_argument(
        "-unf",
        "--unfold",
        type=bool,
        default=False,
        help="unfold the CDO (default is False)",
    )

    # Add observable q-parameter to parser
    parser.add_argument(
        "-obs_q",
        "--obs_q",
        type=int,
        default=2,
        help="observable q-parameter (default is 2)",
    )

    # Send parser to Monte Carlo simulation class and return arguments
    return _parse_mc_args(parser)


# =======================================
# 3. Configuration Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True, slots=True)
class Config:
    # Simulation parameters
    num_times: int = 100
    logtime_i: float = -1.0  # base = dim
    logtime_f: float = 1.5  # base = dim
    unf_logtime_i: float = -1.5  # base = dim
    unf_logtime_f: float = 0.5
    filename: str = "cdo_evolution"

    def _construct_observable(self, N: int, dtype: np.dtype, q: int) -> np.ndarray:
        """Construct observable matrix based on type and number of particles."""
        # Calculate dimension of observable matrix
        dim = 2 ** (N // 2 - 1)

        # Create tuple of Majorana pair operators
        majorana_pairs = create_majorana_pairs(N)

        # Initialize observable matrix
        observable = np.zeros((dim, dim), dtype=dtype, order="F")

        # Retrieve indices for observable terms
        indices = tuple(itertools.combinations(range(N), q))

        # Loop over indices and fill observable matrix
        for idx_tuple in indices:
            # Divide indices into pairs
            pairs = tuple((idx_tuple[i], idx_tuple[i + 1]) for i in range(0, q, 2))

            # Start q-body operator with first pair
            j0, k0 = pairs[0]
            q_body = majorana_pairs[j0][k0]

            # Multiply q-body operator with remaining pairs
            for j, k in pairs[1:]:
                q_body = q_body.dot(majorana_pairs[j][k])

            # Store q-body operator as COO matrix
            q_coo = q_body[:dim, :dim].tocoo()

            # Add q-body operator to observable matrix
            observable[q_coo.row, q_coo.col] += q_coo.data

        # Scale observable by necessary factors for hermicity and extensivity
        observable *= 1j ** (q * (q - 1) / 2) * np.sqrt(N / comb(N, q)) / np.log(2)

        # Return observable matrix
        return observable

    def _construct_initial_state(self, N: int, dtype: np.dtype, q: int) -> np.ndarray:
        """Construct initial state as eigenstate of observable with largest eigenvalue."""
        # Construct observable matrix
        observable = self._construct_observable(N, dtype, q)

        # Calculate eigenvectors of observable
        eigvals, eigvecs = eigh(observable, overwrite_a=True, check_finite=False)

        # Calculate indices of sorted eigenvalues
        sorted_indices = np.argsort(eigvals)

        # Sort eigenvectors based on sorted eigenvalues
        eigvecs = eigvecs[:, sorted_indices]

        # Return last eigenvector as initial state
        return eigvecs[:, -1]

    def _create_times_array(self, base: float) -> np.ndarray:
        """Create times array."""
        return np.logspace(
            self.logtime_i, self.logtime_f, self.num_times, base=base, dtype=np.float64
        )

    def _create_unf_times_array(self, base: float) -> np.ndarray:
        """Create unfolded times array."""
        return np.logspace(
            self.unf_logtime_i,
            self.unf_logtime_f,
            self.num_times,
            base=base,
            dtype=np.float64,
        )


# =======================================
# 4. Simulation Class
# =======================================
@dataclass(repr=False, eq=False, kw_only=True, slots=True)
class CDOEvolution(MonteCarlo):
    # Observable q-parameter
    obs_q: int = 2

    # Unfolding flag
    unfold: bool = False

    # Configuration
    config: Config = field(default_factory=Config)

    # Amount of shared memory to allocate for simulation in bytes
    _shm_size: int = field(init=False, default=0)

    # Shared memory block for evolved states
    _shm: shared_memory.SharedMemory = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Call parent class post-init
        super(CDOEvolution, self).__post_init__()

        # Calculate amount of shared memory required in bytes
        itemsize = np.dtype(self.dtype).itemsize
        self._shm_size = itemsize * self.config.num_times * self.realizs * self.dim

    def run(self) -> None:
        """Run the CDO evolution simulation."""
        # Start timer
        start_time = time()

        # Create shared memory block for evolved states
        self._shm = shared_memory.SharedMemory(create=True, size=self._shm_size)

        # Create arguments for workers
        worker_args = self._create_worker_args()

        # Realize evolved states in parallel
        with Pool(processes=self.workers) as pool:
            pool.map(_calc_evolved_states, worker_args)

        # Calculate CDO statistics in parallel
        with Pool(processes=self.workers) as pool:
            cdo_stats = pool.map(_calc_cdo_stats, worker_args)

        # Close shared memory block
        self._shm.close()

        # Unlink shared memory block
        self._shm.unlink()

        # Process CDO statistics
        self._process_cdo_stats(cdo_stats)

        # End timer and print elapsed time
        elapsed_time = time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    def _check_mc(self) -> None:
        """Check if CDOEvolution simulation is valid."""
        # Call parent class check
        super(CDOEvolution, self)._check_mc()

        # Check if observable q-parameter is valid
        if (
            not isinstance(self.obs_q, int)
            or self.N < self.obs_q < 1
            or self.obs_q % 2 != 0
        ):
            raise ValueError("Observable q-parameter must be an even positive integer.")

        # Check if unfolding flag is valid
        if not isinstance(self.unfold, bool):
            raise ValueError("Unfolding flag must be a boolean value.")

    def _create_worker_args(self) -> list[dict[str, Any]]:
        """Create arguments for workers."""
        # Create dict representation of ensemble
        ens_args = self.ensemble._to_dict_str()

        # Create worker arguments as list of dictionaries
        worker_args = [
            {
                "worker_id": i,
                "num_workers": self.workers,
                "ens_args": ens_args,
                "unfold": self.unfold,
                "obs_q": self.obs_q,
                "realizs": self.realizs,
                "config": asdict(self.config),
                "shm_name": self._shm.name,
            }
            for i in range(self.workers)
        ]

        # Return list of worker arguments
        return worker_args

    def _process_cdo_stats(self, cdo_stats: list[dict[np.ndarray]]) -> None:
        """Process CDO statistics from workers."""
        # Vertically stack CDO statistics from all workers
        cdo_stats = np.vstack(cdo_stats)

        # Store data file name based on unfolding flag
        if self.unfold:
            path = f"{self.output_dir}/{self.config.filename}_unfolded.npz"
        else:
            path = f"{self.output_dir}/{self.config.filename}.npz"

        # Construct observable matrix
        observable = self.config._construct_observable(
            self.ensemble.N, self.ensemble.dtype, self.obs_q
        )

        # Calculate eigenvalues of observable
        eigvals = eigvalsh(observable, overwrite_a=True, check_finite=False)

        # Save CDO statistics to compressed file
        np.savez_compressed(
            path,
            times=cdo_stats[:, 0],
            probabilities=cdo_stats[:, 1:3],
            quantum_purity=cdo_stats[:, 3],
            classical_purity=cdo_stats[:, 4],
            entropy=cdo_stats[:, 5],
            obs_eigvals=eigvals,
            obs_expectation=cdo_stats[:, 6],
            obs_second_moment=cdo_stats[:, 7],
        )

        # Initialize plot of probabilities
        plot = ProbabilitiesPlot(data_path=path, unfold=self.unfold)

        # Plot probabilities
        plot.plot()

        # Initialize plot of purity
        plot = PurityPlot(data_path=path, unfold=self.unfold)

        # Plot purity
        plot.plot()

        # Initialize plot of entropy
        plot = EntropyPlot(data_path=path, unfold=self.unfold)

        # Plot entropy
        plot.plot()

        # Initialize plot of observable expectation value
        plot = ObservablePlot(
            data_path=path,
            unfold=self.unfold,
            obs_q=self.obs_q,
        )

        # Plot observable expectation value
        plot.plot()

    def _worker_memory(self) -> float:
        """Calculate the memory required for each calculation in GiB."""
        # Amount of memory required to store a random matrix
        matrix_memory = self.ensemble.matrix_memory

        # Amount of residual memory required to generate matrices
        resid_memory = self.ensemble.resid_memory

        # Amount of workspace memory required for each calculation
        work_memory = 36 * self.ensemble.dtype.itemsize * self.ensemble.dim

        # Return the total memory required for each worker in GiB
        return (matrix_memory + resid_memory + work_memory) / 2**30

    def _workspace_memory(self) -> float:
        """Calculate the actual workspace memory available in GiB."""
        # Return allocated memory minus shared memory size in GiB
        return self.max_memory - self._shm_size / 2**30


# =======================================
# 5. Main Function
# =======================================
def main() -> None:
    # Create argument parser
    parser = ArgumentParser(description="CDO Evolution Monte Carlo Simulation")

    # Parse command line arguments
    mc_args = _parse_cdo_mc_args(parser)

    # Create an instance of the CDOEvolution class
    mc = CDOEvolution(**mc_args)

    # Run simulation
    mc.run()


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Avoid spawning issues on Windows
    set_start_method("fork", force=True)

    # Configure matplotlib
    configure_matplotlib()

    # Run main function
    main()
