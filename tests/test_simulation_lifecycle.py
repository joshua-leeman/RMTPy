# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import attrs
import numpy as np

from rmtpy.compounds import Compound
from rmtpy.conversion import to_json_compatible
from rmtpy.ensembles import GOE
from rmtpy.simulations.histogram import Histogram
from rmtpy.simulations.histogram2D import Histogram2D
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsSimulation,
)
from rmtpy.simulations.resonance_statistics import ResonanceStatisticsSimulation
from rmtpy.simulations.resonance_statistics.complex_energy_histogram import (
    ComplexEnergyHistogramPlot,
)
from rmtpy.simulations.spectral_statistics import (
    SpectralStatisticsSimulation,
    load_spectral_statistics_result,
    save_spectral_statistics_result,
)
from rmtpy.simulations.spectral_statistics.spectral_histogram import (
    SpectralHistogramPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_statistics_io import (
    plot_spectral_statistics_result,
)
from rmtpy.simulations.time_delay_statistics import TimeDelayStatisticsSimulation


class SimulationLifecycleTests(unittest.TestCase):
    def test_spectral_result_has_explicit_data_order(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=123,
            ),
            realizs=1,
        )

        result = simulation.execute()
        self.assertEqual(
            tuple(data.file_name for data in tuple(result.iterate_data())[:5]),
            (
                "spectral_coeff_1_histogram_data",
                "spectral_coeff_2_histogram_data",
                "spectral_histogram_data",
                "spacings_histogram_data",
                "spectral_form_factors_data",
            ),
        )
        self.assertEqual(len(result.average_by_degree), 1)
        self.assertEqual(result.average_by_degree[0].degree, 2)

    def test_run_context_is_derived_from_attrs_init_fields(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=Compound(ensemble=GOE(num_majoranas=4, seed=123)),
            realizs=2,
            energies=(-0.25, 0.125),
        )

        result = simulation.execute()
        args = result.context.simulation_config["parameters"]
        self.assertEqual(args["realizs"], 2)
        self.assertEqual(args["energies"], [-0.25, 0.125])
        self.assertEqual(args["compound"]["type"], "Compound")
        self.assertEqual(
            args["compound"]["parameters"]["coupling_strengths"],
            simulation.compound.coupling_strengths.tolist(),
        )

    def test_exact_energy_identity_rejects_only_duplicates(self) -> None:
        compound = Compound(ensemble=GOE(num_majoranas=4))

        simulation = TimeDelayStatisticsSimulation(
            compound=compound,
            realizs=1,
            energies=(0.123456, 0.123457),
        )
        np.testing.assert_array_equal(simulation.energies, [0.123456, 0.123457])

        for energies in ((-0.0, 0.0), (0.1, 0.1)):
            with self.subTest(energies=energies), self.assertRaises(ValueError):
                TimeDelayStatisticsSimulation(
                    compound=compound,
                    realizs=1,
                    energies=energies,
                )

    def test_energy_input_is_copied_and_read_only(self) -> None:
        source = np.array([-0.25, 0.25])
        simulation = TimeDelayStatisticsSimulation(
            compound=Compound(ensemble=GOE(num_majoranas=4)),
            realizs=1,
            energies=source,
        )
        source[0] = -0.5

        np.testing.assert_array_equal(simulation.energies, np.array([-0.25, 0.25]))
        self.assertFalse(np.shares_memory(source, simulation.energies))
        self.assertFalse(simulation.energies.flags.writeable)
        with self.assertRaises(ValueError):
            simulation.energies[0] = 0.0

    def test_plotting_does_not_advance_the_live_rng(self) -> None:
        ensemble = GOE(num_majoranas=4, seed=902)
        control = GOE(num_majoranas=4, seed=902)
        simulation = SpectralStatisticsSimulation(ensemble=ensemble, realizs=1)
        control_simulation = SpectralStatisticsSimulation(
            ensemble=control,
            realizs=1,
        )
        result = simulation.execute()
        control_simulation.execute()

        with patch.object(SpectralHistogramPlot, "finish_plot"):
            plot_spectral_statistics_result(
                result,
                out_dir=Path("unused"),
                views="spectral_histogram",
            )

        np.testing.assert_allclose(
            next(ensemble.eigvals_stream(1)),
            next(control.eigvals_stream(1)),
        )

    def test_half_open_2d_bins(self) -> None:
        histogram = Histogram(file_name="example", support=(0.0, 1.0))
        self.assertIsInstance(histogram.support, tuple)

        histogram2d = Histogram2D(
            file_name="example_2d",
            x_support=(0.0, 1.0),
            y_support=(0.0, 1.0),
            x_num_bins=2,
            y_num_bins=2,
        )
        histogram2d.add_histogram_contribution(
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
        )
        self.assertEqual(int(np.sum(histogram2d.counts)), 2)
        self.assertEqual(histogram2d.counts[0, 0], 1)
        self.assertEqual(histogram2d.counts[1, 1], 1)

    def test_degree_zero_experiments_realize_and_finalize(self) -> None:
        spectral = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=0,
                seed=10,
            ),
            realizs=1,
        )
        compound = Compound(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=0,
                seed=20,
            )
        )
        spectral_result = spectral.execute()
        for data in spectral_result.iterate_data():
            for field in attrs.fields(type(data)):
                value = getattr(data, field.name)
                if isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype,
                    np.inexact,
                ):
                    self.assertTrue(np.all(np.isfinite(value)))

        resonance_result = ResonanceStatisticsSimulation(
            compound=compound,
            realizs=1,
        ).execute()
        for data in resonance_result.iterate_data():
            for field in attrs.fields(type(data)):
                value = getattr(data, field.name)
                if isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype,
                    np.inexact,
                ):
                    self.assertTrue(np.all(np.isfinite(value)))

        time_delay_result = TimeDelayStatisticsSimulation(
            compound=compound,
            realizs=1,
            energies=(-0.1, 0.1),
        ).execute()
        for data in time_delay_result.iterate_data():
            for field in attrs.fields(type(data)):
                value = getattr(data, field.name)
                if isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype,
                    np.inexact,
                ):
                    self.assertTrue(np.all(np.isfinite(value)))

        partial_widths_result = PartialWidthsStatisticsSimulation(
            compound=compound,
            realizs=1,
        ).execute()
        for data in partial_widths_result.iterate_data():
            for field in attrs.fields(type(data)):
                value = getattr(data, field.name)
                if isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype,
                    np.inexact,
                ):
                    self.assertTrue(np.all(np.isfinite(value)))

    def test_empty_complex_energy_histogram_can_be_plotted(self) -> None:
        compound = Compound(ensemble=GOE(num_majoranas=4, seed=201))
        histogram = Histogram2D(
            file_name="empty_complex_energy",
            x_support=(-1.0, 1.0),
            y_support=(-4.0, 4.0),
            y_log_base=10.0,
        )
        histogram.compute_histogram_probabilities()
        plot = ComplexEnergyHistogramPlot(
            data=histogram,
            simulation_parameters={"compound": to_json_compatible(compound)},
        )

        with patch.object(ComplexEnergyHistogramPlot, "finish_plot"):
            plot.plot(path=Path("unused"))

    def test_execute_save_load_and_plot_are_explicit(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=0,
                seed=300,
            ),
            realizs=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = simulation.execute()
            run_dir = save_spectral_statistics_result(result, out_dir=tmp_dir)
            loaded = load_spectral_statistics_result(run_dir)
            with patch.object(SpectralHistogramPlot, "plot", autospec=True) as plot:
                plot_spectral_statistics_result(
                    loaded,
                    out_dir=Path(tmp_dir) / "plots",
                    views="spectral_histogram",
                )

            self.assertEqual(run_dir.name, run_dir.name.lower())
            self.assertEqual(len(run_dir.name), 64)
            self.assertEqual(
                len(tuple(run_dir.rglob("*.npz"))),
                len(tuple(result.iterate_data())),
            )
            plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
