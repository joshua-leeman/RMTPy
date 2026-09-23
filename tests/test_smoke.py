# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import unittest
from typing import Any

import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.conversion import RMT_CONVERTER
from rmtpy.ensembles import GaussianOrthogonalEnsemble, ManyBodyEnsemble
from rmtpy.simulations.histogram import Histogram
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsSimulation,
)
from rmtpy.simulations.resonance_statistics import ResonanceStatisticsSimulation
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation
from rmtpy.simulations.time_delay_statistics import TimeDelayStatisticsSimulation
from rmtpy.universal import time_delay_pdf


class SmokeTests(unittest.TestCase):
    def test_ensemble_round_trip(self) -> None:
        ensemble = GaussianOrthogonalEnsemble(num_majoranas=4, seed=123)
        payload: dict[str, Any] = RMT_CONVERTER.unstructure(ensemble)
        restored = RMT_CONVERTER.structure(payload, ManyBodyEnsemble)

        self.assertIsInstance(restored, GaussianOrthogonalEnsemble)
        self.assertEqual(restored.dimension, ensemble.dimension)

    def test_histogram_finalization_is_data_only(self) -> None:
        histogram = Histogram(file_name="example", support=(0.0, 1.0), num_bins=4)
        histogram.add_histogram_contribution(np.array([0.1, 0.2, 0.8]))
        histogram.compute_histogram()

        self.assertEqual(int(np.sum(histogram.counts)), 3)
        self.assertAlmostEqual(
            float(np.sum(histogram.histogram * np.diff(histogram.bins))),
            1.0,
        )

    def test_time_delay_pdf_normalizes(self) -> None:
        num_channels = 4
        heisenberg_time = 7.0
        tau_min = (3 - np.sqrt(8)) / num_channels
        tau_max = (3 + np.sqrt(8)) / num_channels
        times = np.geomspace(
            tau_min * heisenberg_time,
            tau_max * heisenberg_time,
            10_000,
        )

        pdf = time_delay_pdf(
            times,
            num_channels=num_channels,
            heisenberg_time=heisenberg_time,
        )

        self.assertAlmostEqual(np.trapezoid(pdf, times), 1.0, places=4)

    def test_multi_channel_time_delays_are_positive(self) -> None:
        ensemble = GaussianOrthogonalEnsemble(num_majoranas=8, seed=123)
        compound = CompoundEnsemble(ensemble=ensemble, num_free_complex_fermions=1)
        time_delays, _ = next(
            compound.time_delays_stream(energies=np.array([0.0]), realizs=1)
        )

        self.assertGreater(np.min(time_delays), -1e-10)

    def test_statistics_simulations_return_typed_results(self) -> None:
        ensemble = GaussianOrthogonalEnsemble(
            num_majoranas=4,
            max_spectral_polynomial_degree=2,
            seed=123,
        )
        compound = CompoundEnsemble(ensemble=ensemble)

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
            energies=(0.0, 0.1),
        )
        default_time_delay = TimeDelayStatisticsSimulation(
            compound=compound,
            realizs=1,
        )

        np.testing.assert_allclose(default_time_delay.energies, np.array([0.0]))

        spectral_result = spectral.execute()
        resonance_result = resonance.execute()
        partial_widths_result = partial_widths.execute()
        time_delay_result = time_delay.execute()
        self.assertGreater(len(tuple(spectral_result.iterate_data())), 0)
        self.assertGreater(len(tuple(resonance_result.iterate_data())), 0)
        self.assertGreater(len(tuple(partial_widths_result.iterate_data())), 0)
        self.assertGreater(len(tuple(time_delay_result.iterate_data())), 0)

        self.assertEqual(
            spectral_result.coefficients[0].metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            spectral_result.raw.levels.metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            spectral_result.weight.levels.metadata["unfolding"],
            "weight",
        )
        self.assertEqual(
            spectral_result.average_by_degree[0].statistics.levels.metadata["unfolding"],
            "average",
        )
        self.assertEqual(
            spectral_result.variate_by_degree[0].statistics.levels.metadata["unfolding"],
            "variate",
        )

        self.assertEqual(
            resonance_result.coefficients[0].metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            resonance_result.raw.widths.metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            resonance_result.weight.widths.metadata["unfolding"],
            "weight",
        )
        self.assertEqual(
            resonance_result.average_by_degree[0].statistics.complex_energies.metadata[
                "unfolding"
            ],
            "average",
        )

        self.assertEqual(
            partial_widths_result.statistics[0].histogram.metadata["unfolding"],
            "raw",
        )

        self.assertEqual(
            time_delay_result.raw[0].histogram.file_name,
            "time_delay_histogram_data",
        )
        self.assertEqual(time_delay_result.raw[1].energy_index, 1)
        self.assertEqual(time_delay_result.raw[1].energy, 0.1)
        self.assertEqual(
            time_delay_result.raw[0].histogram.metadata["unfolding"],
            "raw",
        )
        self.assertEqual(
            len(time_delay_result.raw),
            2,
        )
        self.assertIsInstance(
            time_delay.energies,
            np.ndarray,
        )
        self.assertEqual(
            time_delay_result.context.output_request["unfolding_modes"],
            ["raw", "weight", "average", "variate"],
        )


if __name__ == "__main__":
    unittest.main()
