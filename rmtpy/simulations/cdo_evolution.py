# rmtpy.simulations.cdo_evolution.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, set_start_method, shared_memory
from time import time
from typing import Any

# Third-party imports
import numpy as np
from scipy.linalg import eigvalsh
from scipy.special import jn_zeros

# Local application imports
from rmtpy.plotting.cdo_evolution.probabilities_plot import ProbabilitiesPlot
from rmtpy.plotting.cdo_evolution.purity_plot import PurityPlot
from rmtpy.plotting.cdo_evolution.entropy_plot import EntropyPlot
from rmtpy.simulations._mc import MonteCarlo, _parse_mc_args
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


def cdo_expectation(cdo: np.ndarray, observable: np.ndarray) -> np.ndarray:
    """Calculate expectation value of observable from CDO."""
    # Right-multiply CDO with observable
    products = cdo @ observable

    # Evaluate trace of each product and return it
    return np.trace(products).real


def cdo_probabilities(cdo: np.ndarray) -> np.ndarray:
    """Calculate probabilities from CDO."""
    # Return diagonal elements of CDO
    return np.diagonal(cdo).real


def cdo_quantum_purity(cdo: np.ndarray) -> np.ndarray:
    """Calculate purity from CDO."""
    # Compute purity from CDO and return it
    return np.trace(cdo @ cdo).real


def cdo_classic_purity(cdo: np.ndarray) -> np.ndarray:
    """Calculate classical purity from CDO."""
    # Compute population purity from CDO and return it
    return np.sum(cdo_probabilities(cdo) ** 2).real


def cdo_entropy(cdo: np.ndarray, overwrite: bool = False) -> np.ndarray:
    """Calculate von Neumann entropy from CDO."""
    # Compute eigenvalues of CDO
    eigvals = eigvalsh(cdo, overwrite_a=overwrite, check_finite=False)

    # Clip eigenvalues to avoid numerical issues
    eigvals = eigvals[eigvals > 1e-10]

    # Compute entropy from eigenvalues and return it
    return -np.sum(eigvals * np.log(eigvals))


def _calc_evolved_states(args: dict[str, Any]) -> dict[np.ndarray]:
    """Worker function to calculate evolved states."""
    # Unpack ensemble arguments
    ens_args = args["ens_args"]

    # Find and initialize ensemble
    ensemble = get_ensemble(ens_args)

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

    # Declare initial state
    initial_state = np.zeros(ensemble.dim, dtype=ensemble.dtype)
    initial_state[0] = 1.0

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
    evolved_states = np.ndarray(
        (config.num_times, realizs, ensemble.dim),
        dtype=ensemble.dtype,
        buffer=shm.buf,
        order="F",
    )

    # Initialize array to store CDO statistics
    cdo_stats = np.empty((len(times), 6), dtype=ensemble.real_dtype)

    # Store times in first column of cdo_stats
    cdo_stats[:, 0] = times

    # Loop over times, calculate CDO, and perform statistics
    for i, t in enumerate(range(start, end)):
        # Calculate CDO
        cdo = calculate_cdo(evolved_states[t, :, :])

        # Calculate probabilities
        cdo_stats[i, 1:3] = cdo_probabilities(cdo)[:2]

        # Calculate quantum purity and remove bias
        q_purity = cdo_quantum_purity(cdo)
        cdo_stats[i, 3] = (realizs * q_purity - 1) / (realizs - 1)

        # Calculate classical purity and remove bias
        c_purity = cdo_classic_purity(cdo)
        cdo_stats[i, 4] = (realizs * c_purity - 1) / (realizs - 1)

        # Calculate entropy and remove bias
        entropy = cdo_entropy(cdo)
        cdo_stats[i, 5] = entropy + (ensemble.dim - 1) / 2 / realizs

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

    # Send parser to Monte Carlo simulation class and return arguments
    return _parse_mc_args(parser)


# =======================================
# 3. Configuration Dataclass
# =======================================
@dataclass(repr=False, eq=False, kw_only=True, slots=True)
class Config:
    # Simulation parameters
    num_times: int = 1000
    logtime_i: float = -0.5  # base = dim
    logtime_f: float = 1.5  # base = dim
    unf_logtime_i: float = -1.5  # base = dim
    unf_logtime_f: float = 0.5
    filename: str = "cdo_evolution"

    def _construct_observable(self) -> np.ndarray:
        pass

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
    # Unfolding flag
    unfold: bool = field(default=False)

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

        # Save CDO statistics to compressed file
        np.savez_compressed(
            path,
            times=cdo_stats[:, 0],
            probabilities=cdo_stats[:, 1:3],
            quantum_purity=cdo_stats[:, 3],
            classical_purity=cdo_stats[:, 4],
            entropy=cdo_stats[:, 5],
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
