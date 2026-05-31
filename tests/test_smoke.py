import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from rmtpy.compounds import Compound
from rmtpy.conversion import RMT_CONVERTER
from rmtpy.ensembles import GaussianOrthogonalEnsemble, ManyBodyEnsemble
from rmtpy.simulations.histogram import Histogram, finalize_histogram
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsSimulation,
)
from rmtpy.simulations.resonance_statistics import ResonanceStatisticsSimulation
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation
from rmtpy.simulations.time_delay_statistics import TimeDelayStatisticsSimulation


class SmokeTests(unittest.TestCase):
    def test_ensemble_round_trip(self) -> None:
        ensemble = GaussianOrthogonalEnsemble(num_majoranas=4, seed=123)
        payload: dict[str, Any] = RMT_CONVERTER.unstructure(ensemble)
        restored = RMT_CONVERTER.structure(payload, ManyBodyEnsemble)

        self.assertIsInstance(restored, GaussianOrthogonalEnsemble)
        self.assertEqual(restored.dimension, ensemble.dimension)

    def test_histogram_save_load_round_trip(self) -> None:
        histogram = Histogram(file_name="example", support=(0.0, 1.0), num_bins=4)
        histogram.add_histogram_contribution(np.array([0.1, 0.2, 0.8]))
        finalize_histogram(histogram)

        with tempfile.TemporaryDirectory() as tmpdir:
            path: Path = Path(tmpdir) / "example_data.npz"
            histogram.save(path)
            restored: Histogram = Histogram.load(path)

        np.testing.assert_array_equal(restored.counts, histogram.counts)
        np.testing.assert_allclose(restored.histogram, histogram.histogram)

    def test_statistics_simulations_construct_observables(self) -> None:
        ensemble = GaussianOrthogonalEnsemble(
            num_majoranas=4,
            max_spectral_polynomial_degree=2,
            seed=123,
        )
        compound = Compound(ensemble=ensemble)

        spectral = SpectralStatisticsSimulation(
            ensemble=ensemble,
            realizs=1,
        )
        resonance = ResonanceStatisticsSimulation(
            compound=compound,
            realizs=1,
        )
        partial_widths = PartialWidthsStatisticsSimulation(
            compound=compound,
            realizs=1,
        )
        time_delay = TimeDelayStatisticsSimulation(
            compound=compound,
            realizs=1,
            energies=[0.0, 0.1],
        )
        default_time_delay = TimeDelayStatisticsSimulation(
            compound=compound,
            realizs=1,
        )

        np.testing.assert_allclose(default_time_delay.energies, np.array([0.0]))

        self.assertGreater(len(tuple(spectral.iter_observables())), 0)
        self.assertGreater(len(tuple(resonance.iter_observables())), 0)
        self.assertGreater(len(tuple(partial_widths.iter_observables())), 0)
        self.assertGreater(len(tuple(time_delay.iter_observables())), 0)

        spe_out = spectral.outputs
        res_out = resonance.outputs
        wid_out = partial_widths.outputs
        tim_out = time_delay.outputs

        self.assertEqual(
            spe_out.coefficients.by_degree[0].metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            spe_out.raw.levels.metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            spe_out.weight_unfolded.levels.metadata["unfolding"],
            "weight",
        )
        self.assertEqual(
            spe_out.avg_unfolded_by_degree[0].levels.metadata["unfolding"],
            "avg",
        )
        self.assertEqual(
            spe_out.var_unfolded_by_degree[0].levels.metadata["unfolding"],
            "var",
        )

        self.assertEqual(
            res_out.coefficients.by_degree[0].metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            res_out.raw.widths.metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            res_out.weight_unfolded.widths.metadata["unfolding"],
            "weight",
        )
        self.assertEqual(
            res_out.avg_unfolded_by_degree[0].complex_energies.metadata["unfolding"],
            "avg",
        )

        self.assertEqual(
            wid_out.histograms[0].metadata["unfolding"],
            "raw",
        )

        self.assertEqual(
            tim_out.raw[0].data.file_name,
            "time_delay_histogram_data",
        )
        self.assertEqual(
            time_delay.observable_output_path(tim_out.raw[1]),
            Path("energy_0p1"),
        )
        self.assertEqual(
            tim_out.raw[0].metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            len(tim_out.raw),
            2,
        )
        self.assertIsInstance(
            time_delay.energies,
            np.ndarray,
        )
        self.assertNotIn(
            "energies_",
            str(time_delay.to_path),
        )


if __name__ == "__main__":
    unittest.main()
