import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import attrs
import numpy as np

from rmtpy.compounds import Compound
from rmtpy.ensembles import GOE, ManyBodyEnsemble
from rmtpy.simulations.histogram import Histogram
from rmtpy.simulations.histogram2D import Histogram2D
from rmtpy.simulations.observable import Observable
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsSimulation,
)
from rmtpy.simulations.plot import plot_data
from rmtpy.simulations.resonance_statistics import ResonanceStatisticsSimulation
from rmtpy.simulations.resonance_statistics.complex_energy_histogram import (
    ComplexEnergyHistogramPlot,
)
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation
from rmtpy.simulations.spectral_statistics.spectral_histogram import (
    SpectralHistogramPlot,
)
from rmtpy.simulations.time_delay_statistics import TimeDelayStatisticsSimulation


class SimulationLifecycleTests(unittest.TestCase):
    def test_explicit_output_order_and_public_lookup(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=123,
            ),
            realizs=1,
        )

        observables = tuple(simulation.iter_observables())
        self.assertEqual(
            tuple(observable.data.file_name for observable in observables[:5]),
            (
                "spectral_coeff_1_histogram_data",
                "spectral_coeff_2_histogram_data",
                "spectral_histogram_data",
                "spacings_histogram_data",
                "spectral_form_factors_data",
            ),
        )
        self.assertIs(
            simulation.get_data("spectral_histogram"),
            simulation.outputs.raw.levels.data,
        )
        self.assertEqual(
            len(simulation.find_observables(unfolding="avg", degree=2)),
            3,
        )
        with self.assertRaises(LookupError):
            simulation.get_observable(unfolding="raw")

    def test_metadata_is_derived_from_attrs_init_fields(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=Compound(ensemble=GOE(num_majoranas=4, seed=123)),
            realizs=2,
            energies=(-0.25, 0.125),
        )

        args = simulation.metadata["args"]
        self.assertEqual(args["realizs"], 2)
        self.assertEqual(args["energies"], [-0.25, 0.125])
        self.assertEqual(args["compound"]["name"], "compound")
        self.assertEqual(
            args["compound"]["args"]["coupling_strengths"],
            simulation.compound.coupling_strengths.tolist(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation.save_metadata(tmp_dir)
            with open(Path(tmp_dir) / "metadata.json") as file:
                saved_metadata = json.load(file)
        self.assertEqual(
            saved_metadata["args"]["compound"]["args"]["coupling_strengths"],
            simulation.compound.coupling_strengths.tolist(),
        )

    def test_energy_path_collisions_are_rejected(self) -> None:
        compound = Compound(ensemble=GOE(num_majoranas=4))

        for energies in ((0.123456, 0.123457), (-0.0, 0.0), (0.1, 0.1)):
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

    def test_observables_share_one_transient_plot_model(self) -> None:
        ensemble = GOE(num_majoranas=4, seed=123)
        simulation = SpectralStatisticsSimulation(ensemble=ensemble, realizs=1)
        observable = simulation.outputs.raw.levels

        self.assertFalse(hasattr(observable, "plot"))
        runtime_args = deepcopy(simulation.metadata["args"])
        first_plot = observable.plot_cls(
            data=observable.data,
            runtime_simulation_args=runtime_args,
        )
        second_plot = simulation.outputs.raw.spacings.plot_cls(
            data=simulation.outputs.raw.spacings.data,
            runtime_simulation_args=runtime_args,
        )
        first_model = first_plot.structure_simulation_arg(
            "ensemble",
            ManyBodyEnsemble,
        )
        second_model = second_plot.structure_simulation_arg(
            "ensemble",
            ManyBodyEnsemble,
        )

        self.assertIs(first_model, second_model)
        self.assertIsNot(first_model, ensemble)

    def test_plotting_does_not_advance_the_live_rng(self) -> None:
        ensemble = GOE(num_majoranas=4, seed=902)
        control = GOE(num_majoranas=4, seed=902)
        simulation = SpectralStatisticsSimulation(ensemble=ensemble, realizs=1)
        observable = simulation.outputs.raw.levels

        with patch.object(SpectralHistogramPlot, "finish_plot"):
            observable.save_plot(
                Path("unused"),
                simulation_args=deepcopy(simulation.metadata["args"]),
            )

        np.testing.assert_allclose(
            next(ensemble.eigvals_stream(1)),
            next(control.eigvals_stream(1)),
        )

    def test_archive_members_and_half_open_2d_bins(self) -> None:
        histogram = Histogram(file_name="example", support=(0.0, 1.0))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "example_data.npz"
            histogram.save(path)
            with np.load(path, allow_pickle=True) as archive:
                self.assertNotIn("allow_pickle", archive.files)
            restored = Histogram.load(path)

        self.assertIsInstance(restored.support, tuple)
        self.assertEqual(restored.support, histogram.support)

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
        simulations = (
            spectral,
            ResonanceStatisticsSimulation(compound=compound, realizs=1),
            PartialWidthsStatisticsSimulation(compound=compound, realizs=1),
            TimeDelayStatisticsSimulation(
                compound=compound,
                realizs=1,
                energies=(-0.1, 0.1),
            ),
        )

        for simulation in simulations:
            with self.subTest(simulation=type(simulation).__name__):
                simulation.realize_monte_carlo_simulation()
                simulation.calculate_statistics()
                for observable in simulation.iter_observables():
                    for field in attrs.fields(type(observable.data)):
                        value = getattr(observable.data, field.name)
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
            runtime_simulation_args={"compound": compound},
        )

        with patch.object(ComplexEnergyHistogramPlot, "finish_plot"):
            plot.plot(path=Path("unused"))

    def test_run_persists_outputs_and_saved_plot_fallback(self) -> None:
        simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=0,
                seed=300,
            ),
            realizs=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(Observable, "save_plot", autospec=True) as save_plot:
                simulation.run(out_dir=tmp_dir)

            base_dir = Path(tmp_dir) / simulation.to_path
            archives = tuple(base_dir.rglob("*.npz"))
            self.assertEqual(len(archives), len(tuple(simulation.iter_observables())))
            self.assertEqual(save_plot.call_count, len(archives))

            raw_path = base_dir / "spectral_histogram" / "spectral_histogram_data.npz"
            with patch.object(
                SpectralHistogramPlot,
                "plot",
                autospec=True,
            ) as render:
                plot_data(raw_path, plot_cls=SpectralHistogramPlot)

            render.assert_called_once()
            restored_plot = render.call_args.args[0]
            restored_ensemble = restored_plot.structure_simulation_arg(
                "ensemble",
                ManyBodyEnsemble,
            )
            self.assertIsInstance(restored_ensemble, GOE)
            self.assertEqual(restored_ensemble.num_majoranas, 4)


if __name__ == "__main__":
    unittest.main()
