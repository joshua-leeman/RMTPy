import unittest
from pathlib import Path
from types import ModuleType
from typing import cast

import rmtpy.simulations as simulations
import rmtpy.simulations.partial_widths_statistics as partial_widths_statistics
import rmtpy.simulations.resonance_statistics as resonance_statistics
import rmtpy.simulations.spectral_statistics as spectral_statistics
import rmtpy.simulations.time_delay_statistics as time_delay_statistics
import rmtpy.simulations.transmission_coefficients as transmission_coefficients

SIMULATION_APIS: tuple[tuple[ModuleType, set[str]], ...] = (
    (
        spectral_statistics,
        {
            "SpectralStatisticsSimulation",
            "load_spectral_statistics_simulation",
            "plot_spectral_statistics_simulation",
            "run_spectral_statistics_simulation",
        },
    ),
    (
        partial_widths_statistics,
        {
            "PartialWidthsStatisticsSimulation",
            "load_partial_widths_statistics_simulation",
            "plot_partial_widths_statistics_simulation",
            "run_partial_widths_statistics_simulation",
        },
    ),
    (
        resonance_statistics,
        {
            "ResonanceStatisticsSimulation",
            "load_resonance_statistics_simulation",
            "plot_resonance_statistics_simulation",
            "run_resonance_statistics_simulation",
        },
    ),
    (
        time_delay_statistics,
        {
            "TimeDelayStatisticsSimulation",
            "load_time_delay_statistics_simulation",
            "plot_time_delay_statistics_simulation",
            "run_time_delay_statistics_simulation",
        },
    ),
    (
        transmission_coefficients,
        {
            "TransmissionCoefficientsSimulation",
            "load_transmission_coefficients_simulation",
            "plot_transmission_coefficients_simulation",
            "run_transmission_coefficients_simulation",
        },
    ),
)

OBSOLETE_PUBLIC_NAMES: set[str] = {
    "ChannelTransmissionResult",
    "PartialWidthStatistic",
    "PartialWidthsStatisticsResult",
    "ResonanceStatisticsRequest",
    "ResonanceStatisticsResult",
    "SpectralStatisticsRequest",
    "SpectralStatisticsResult",
    "TimeDelayStatisticsRequest",
    "TimeDelayStatisticsResult",
    "TotalWidthStatistic",
    "TransmissionCoefficientsResult",
    "load_partial_widths_statistics_result",
    "load_resonance_statistics_result",
    "load_spectral_statistics_result",
    "load_time_delay_statistics_result",
    "load_transmission_coefficients_result",
    "plot_partial_widths_statistics_result",
    "plot_resonance_statistics_result",
    "plot_spectral_statistics_result",
    "plot_time_delay_statistics_result",
    "plot_transmission_coefficients_result",
    "run_partial_widths_statistics",
    "run_resonance_statistics",
    "run_spectral_statistics",
    "run_time_delay_statistics",
    "run_transmission_coefficients",
    "save_partial_widths_statistics_result",
    "save_resonance_statistics_result",
    "save_spectral_statistics_result",
    "save_time_delay_statistics_result",
    "save_transmission_coefficients_result",
}


class SimulationsPublicAPITests(unittest.TestCase):
    def test_concrete_packages_export_only_the_canonical_simulation_api(self) -> None:
        for module, expected_api in SIMULATION_APIS:
            with self.subTest(module=module.__name__):
                self.assertEqual(set(module.__all__), expected_api)
                self.assertTrue(all(hasattr(module, name) for name in expected_api))
                self.assertTrue(
                    all(not hasattr(module, name) for name in OBSOLETE_PUBLIC_NAMES)
                )

    def test_parent_package_exports_every_canonical_simulation_api(self) -> None:
        expected_api = {"Simulation", "SimulationManifest"}
        for _, concrete_api in SIMULATION_APIS:
            expected_api.update(concrete_api)

        self.assertEqual(set(simulations.__all__), expected_api)
        self.assertTrue(all(hasattr(simulations, name) for name in expected_api))
        self.assertTrue(
            all(not hasattr(simulations, name) for name in OBSOLETE_PUBLIC_NAMES)
        )

    def test_obsolete_modules_and_package_paths_are_absent(self) -> None:
        simulations_directory = Path(simulations.__file__).parent
        self.assertFalse(
            (simulations_directory / "transmission_coefficients_simulation").exists()
        )
        self.assertFalse(
            (
                simulations_directory / "time_delay_statistics" / "time_delay_histograms"
            ).exists()
        )

        for module, _ in SIMULATION_APIS:
            package_directory = Path(cast(str, module.__file__)).parent
            for obsolete_file_name in (
                "data_factories.py",
                "request.py",
                "result_io.py",
                "results.py",
            ):
                with self.subTest(
                    module=module.__name__,
                    obsolete_file_name=obsolete_file_name,
                ):
                    self.assertFalse((package_directory / obsolete_file_name).exists())


if __name__ == "__main__":
    unittest.main()
