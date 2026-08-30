import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rmtpy.compounds import Compound
from rmtpy.ensembles import GaussianOrthogonalEnsemble
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsSimulation,
)
from rmtpy.simulations.plot import (
    DIMENSION_TIME_LOG_SUPPORT,
    UNFOLDED_DIMENSION_TIME_LOG_SUPPORT,
    DimensionTimeAxes,
    UnfoldedDimensionTimeAxes,
)
from rmtpy.simulations.resonance_statistics import ResonanceStatisticsSimulation
from rmtpy.simulations.resonance_statistics.observables import (
    RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
    UNFOLDED_RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors.resonance_form_factor_plot import (
    ResonanceFormFactorsAxes,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsAxes,
    UnfoldedResonanceFormFactorsPlot,
)
from rmtpy.simulations.spectral_statistics.observables import (
    SFF_LOG_D_TIME_SUPPORT,
    UNFOLDED_SFF_LOG_D_TIME_SUPPORT,
)
from rmtpy.simulations.spectral_statistics.spectral_form_factors.spectral_form_factors_plot import (
    FormFactorsAxes,
    FormFactorsPlot,
    UnfoldedFormFactorsAxes,
    UnfoldedFormFactorsPlot,
)
from rmtpy.simulations.time_delay_statistics import TimeDelayStatisticsSimulation
from rmtpy.simulations.time_delay_statistics.observables import (
    RAW_LOG_D_TIME_DELAY_SUPPORT,
    UNFOLDED_LOG_D_TIME_DELAY_SUPPORT,
)
from rmtpy.simulations.time_delay_statistics.time_delay_histograms.time_delay_histograms_plot import (
    TimeDelayHistogramAxes,
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramAxes,
    UnfoldedTimeDelayHistogramPlot,
)
from rmtpy.simulations.transmission_coefficients_simulation import (
    TransmissionCoefficientsSimulation,
)


def create_compound(
    *,
    max_polynomial_degree: int,
    num_free_complex_fermions: int = 1,
    seed: int = 123,
) -> Compound:
    ensemble = GaussianOrthogonalEnsemble(
        num_majoranas=4,
        max_spectral_polynomial_degree=max_polynomial_degree,
        seed=seed,
    )
    return Compound(
        ensemble=ensemble,
        num_free_complex_fermions=num_free_complex_fermions,
    )


def resonance_group_observables(outputs) -> tuple:
    return (
        outputs.resonances,
        outputs.widths,
        outputs.spacings,
        outputs.complex_energies,
        outputs.form_factors,
    )


class SharedDimensionTimePlotTests(unittest.TestCase):
    def test_raw_time_plots_share_axes_and_support(self) -> None:
        for axes_cls in (
            FormFactorsAxes,
            ResonanceFormFactorsAxes,
            TimeDelayHistogramAxes,
        ):
            with self.subTest(axes_cls=axes_cls):
                self.assertTrue(issubclass(axes_cls, DimensionTimeAxes))
                axes = axes_cls()
                self.assertEqual(axes.xticks, DimensionTimeAxes().xticks)
                self.assertEqual(axes.xtick_labels, DimensionTimeAxes().xtick_labels)
                self.assertEqual(axes.xlabel, DimensionTimeAxes().xlabel)

        for plot_cls in (
            FormFactorsPlot,
            ResonanceFormFactorsPlot,
            TimeDelayHistogramPlot,
        ):
            with self.subTest(plot_cls=plot_cls):
                self.assertEqual(plot_cls.xlim, DIMENSION_TIME_LOG_SUPPORT)

        self.assertEqual(SFF_LOG_D_TIME_SUPPORT, DIMENSION_TIME_LOG_SUPPORT)
        self.assertEqual(
            RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
            DIMENSION_TIME_LOG_SUPPORT,
        )
        self.assertEqual(
            RAW_LOG_D_TIME_DELAY_SUPPORT,
            DIMENSION_TIME_LOG_SUPPORT,
        )

    def test_unfolded_time_plots_share_axes_and_support(self) -> None:
        for axes_cls in (
            UnfoldedFormFactorsAxes,
            UnfoldedResonanceFormFactorsAxes,
            UnfoldedTimeDelayHistogramAxes,
        ):
            with self.subTest(axes_cls=axes_cls):
                self.assertTrue(issubclass(axes_cls, UnfoldedDimensionTimeAxes))
                axes = axes_cls()
                reference_axes = UnfoldedDimensionTimeAxes()
                self.assertEqual(axes.xticks, reference_axes.xticks)
                self.assertEqual(axes.xtick_labels, reference_axes.xtick_labels)
                self.assertEqual(axes.xlabel, reference_axes.xlabel)

        for plot_cls in (
            UnfoldedFormFactorsPlot,
            UnfoldedResonanceFormFactorsPlot,
            UnfoldedTimeDelayHistogramPlot,
        ):
            with self.subTest(plot_cls=plot_cls):
                self.assertEqual(
                    plot_cls.xlim,
                    UNFOLDED_DIMENSION_TIME_LOG_SUPPORT,
                )

        self.assertEqual(
            UNFOLDED_SFF_LOG_D_TIME_SUPPORT,
            UNFOLDED_DIMENSION_TIME_LOG_SUPPORT,
        )
        self.assertEqual(
            UNFOLDED_RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
            UNFOLDED_DIMENSION_TIME_LOG_SUPPORT,
        )
        self.assertEqual(
            UNFOLDED_LOG_D_TIME_DELAY_SUPPORT,
            UNFOLDED_DIMENSION_TIME_LOG_SUPPORT,
        )


class ResonanceOutputTests(unittest.TestCase):
    def test_group_counts_and_metadata_by_maximum_degree(self) -> None:
        expected = {
            0: ((), 10),
            2: ((2,), 22),
            4: ((2, 4), 34),
        }

        for max_degree, (degrees, observable_count) in expected.items():
            with self.subTest(max_degree=max_degree):
                simulation = ResonanceStatisticsSimulation(
                    compound=create_compound(max_polynomial_degree=max_degree),
                    realizs=1,
                )
                outputs = simulation.outputs

                self.assertEqual(simulation.truncated_degrees, degrees)
                self.assertEqual(
                    len(tuple(simulation.iter_observables())),
                    observable_count,
                )
                self.assertEqual(
                    tuple(
                        observable.metadata["degree"]
                        for observable in outputs.coefficients.by_degree
                    ),
                    tuple(range(1, max_degree + 1)),
                )

                for observable in resonance_group_observables(outputs.raw):
                    self.assertEqual(observable.metadata["unfolding"], "raw")
                    self.assertNotIn("degree", observable.metadata)

                for observable in resonance_group_observables(outputs.weight_unfolded):
                    self.assertEqual(observable.metadata["unfolding"], "wgt")
                    self.assertNotIn("degree", observable.metadata)

                for unfolding, groups in (
                    ("avg", outputs.avg_unfolded_by_degree),
                    ("var", outputs.var_unfolded_by_degree),
                ):
                    self.assertEqual(len(groups), len(degrees))
                    for degree, group in zip(degrees, groups, strict=True):
                        for observable in resonance_group_observables(group):
                            self.assertEqual(
                                observable.metadata["unfolding"],
                                unfolding,
                            )
                            self.assertEqual(observable.metadata["degree"], degree)


class TimeDelayOutputTests(unittest.TestCase):
    def test_degree_by_energy_structure_and_metadata(self) -> None:
        energies = (-0.25, 0.0, 0.125)
        expected_degrees = {
            0: (),
            2: (2,),
            4: (2, 4),
        }

        for max_degree, degrees in expected_degrees.items():
            with self.subTest(max_degree=max_degree):
                simulation = TimeDelayStatisticsSimulation(
                    compound=create_compound(max_polynomial_degree=max_degree),
                    realizs=1,
                    energies=energies,
                )
                outputs = simulation.outputs
                num_energies = len(energies)

                self.assertEqual(simulation.truncated_degrees, degrees)
                self.assertEqual(len(outputs.raw), num_energies)
                self.assertEqual(len(outputs.weight_unfolded), num_energies)
                self.assertEqual(len(outputs.avg_unfolded_by_degree), len(degrees))
                self.assertEqual(len(outputs.var_unfolded_by_degree), len(degrees))
                self.assertEqual(
                    len(tuple(simulation.iter_observables())),
                    2 * num_energies * (len(degrees) + 1),
                )

                for unfolding, group in (
                    ("raw", outputs.raw),
                    ("wgt", outputs.weight_unfolded),
                ):
                    self.assertEqual(len(group), num_energies)
                    for energy_index, (energy, observable) in enumerate(
                        zip(energies, group, strict=True)
                    ):
                        self.assertEqual(observable.metadata["unfolding"], unfolding)
                        self.assertEqual(
                            observable.metadata["energy_index"], energy_index
                        )
                        self.assertEqual(observable.metadata["energy"], energy)
                        self.assertNotIn("degree", observable.metadata)

                for unfolding, groups in (
                    ("avg", outputs.avg_unfolded_by_degree),
                    ("var", outputs.var_unfolded_by_degree),
                ):
                    for degree, group in zip(degrees, groups, strict=True):
                        self.assertEqual(len(group), num_energies)
                        for energy_index, (energy, observable) in enumerate(
                            zip(energies, group, strict=True)
                        ):
                            self.assertEqual(
                                observable.metadata["unfolding"],
                                unfolding,
                            )
                            self.assertEqual(observable.metadata["degree"], degree)
                            self.assertEqual(
                                observable.metadata["energy_index"],
                                energy_index,
                            )
                            self.assertEqual(observable.metadata["energy"], energy)

                raw_paths = tuple(
                    simulation.observable_output_path(observable)
                    for observable in outputs.raw
                )
                self.assertEqual(len(set(raw_paths)), num_energies)

                output_locations = {
                    (
                        simulation.observable_output_path(observable),
                        observable.data.file_name,
                    )
                    for observable in simulation.iter_observables()
                }
                self.assertEqual(
                    len(output_locations),
                    len(tuple(simulation.iter_observables())),
                )


class TransmissionCoefficientOutputTests(unittest.TestCase):
    def test_each_selected_channel_is_averaged_across_energy(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=create_compound(max_polynomial_degree=0),
            realizs=2,
            channel_indices=(0, 1),
        )

        spectral_density = simulation.compound.ensemble.spectral_density
        expected_energy_range = spectral_density.plot_range
        self.assertGreaterEqual(len(simulation.energies), 2)
        np.testing.assert_allclose(
            simulation.energies[[0, -1]],
            expected_energy_range,
        )
        self.assertFalse(simulation.energies.flags.writeable)

        channel_0_diagonal = np.linspace(0.5, 0.75, len(simulation.energies))
        channel_1_diagonal = np.linspace(0.25, 0.0, len(simulation.energies))
        scattering_matrices = np.zeros(
            (len(simulation.energies), 2, 2),
            dtype=np.complex128,
        )
        scattering_matrices[:, 0, 0] = channel_0_diagonal
        scattering_matrices[:, 1, 1] = channel_1_diagonal
        simulation.outputs.add_scattering_matrices(scattering_matrices)
        simulation.outputs.add_scattering_matrices(scattering_matrices)
        simulation.calculate_statistics()

        self.assertEqual(len(simulation.outputs.by_channel), 2)
        expected_coefficients = (
            1.0 - channel_0_diagonal**2,
            1.0 - channel_1_diagonal**2,
        )
        for channel_index, (observable, expected) in enumerate(
            zip(
                simulation.outputs.by_channel,
                expected_coefficients,
                strict=True,
            )
        ):
            self.assertEqual(observable.metadata["channel_index"], channel_index)
            self.assertEqual(observable.data.channel_index, channel_index)
            self.assertEqual(observable.data.realizs, 2)
            np.testing.assert_array_equal(observable.data.energies, simulation.energies)
            np.testing.assert_allclose(
                observable.data.transmission_coefficients,
                expected,
            )

        output_paths = tuple(
            simulation.observable_output_path(observable)
            for observable in simulation.outputs.by_channel
        )
        self.assertEqual(output_paths, (Path("channel_0"), Path("channel_1")))

        observable = simulation.outputs.by_channel[0]
        plot = observable.plot_cls(
            data=observable.data,
            runtime_simulation_args={"compound": simulation.compound},
        )
        with patch.object(plot, "finish_plot"):
            plot.plot(path="unused")

        np.testing.assert_array_equal(plot.ax.lines[0].get_xdata(), simulation.energies)
        np.testing.assert_allclose(plot.xlim, expected_energy_range)
        self.assertEqual(plot.axes.xlabel, r"$E / E_0$")
        self.assertEqual(plot.axes.ylabel, r"$T_{0}(E)$")

    def test_invalid_channel_selections_are_rejected(self) -> None:
        compound = create_compound(max_polynomial_degree=0)
        for channel_indices in ((), (0, 0), (-1,), (compound.num_channels,)):
            with self.subTest(channel_indices=channel_indices), self.assertRaises(
                ValueError
            ):
                TransmissionCoefficientsSimulation(
                    compound=compound,
                    realizs=1,
                    channel_indices=channel_indices,
                )

    def test_weisskopf_estimate_uses_every_open_channel(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=create_compound(max_polynomial_degree=0),
            realizs=2,
            channel_indices=(0,),
        )

        scattering_matrices = np.zeros(
            (len(simulation.energies), 2, 2),
            dtype=np.complex128,
        )
        scattering_matrices[:, 0, 0] = 0.5
        scattering_matrices[:, 1, 1] = 0.0
        simulation.outputs.add_scattering_matrices(scattering_matrices)
        simulation.outputs.add_scattering_matrices(scattering_matrices)
        simulation.calculate_statistics()

        observable = simulation.outputs.weisskopf_estimate
        data = observable.data
        self.assertEqual(data.num_channels, simulation.compound.num_channels)
        self.assertEqual(data.realizs, 2)
        np.testing.assert_allclose(
            data.transmission_coefficients,
            np.tile((0.75, 1.0), (len(simulation.energies), 1)),
        )

        spectral_density = simulation.compound.ensemble.spectral_density
        weight_density = spectral_density.weight_pdf(simulation.energies)
        in_support = weight_density > 0.0
        expected = (
            1.75
            / (2.0 * simulation.compound.ensemble.dimension * weight_density[in_support])
        )
        np.testing.assert_allclose(data.weisskopf_estimate[in_support], expected)
        self.assertTrue(np.all(np.isnan(data.weisskopf_estimate[~in_support])))

        self.assertEqual(
            simulation.observable_output_path(observable),
            Path(),
        )
        self.assertEqual(len(tuple(simulation.iter_observables())), 2)

        plot = observable.plot_cls(
            data=data,
            runtime_simulation_args={"compound": simulation.compound},
        )
        with patch.object(plot, "finish_plot"):
            plot.plot(path="unused")

        np.testing.assert_array_equal(plot.ax.lines[0].get_xdata(), simulation.energies)
        np.testing.assert_allclose(plot.xlim, spectral_density.plot_range)
        self.assertEqual(plot.axes.xlabel, r"$E / E_0$")
        self.assertEqual(
            plot.axes.ylabel,
            r"$\Gamma_{\mathrm{Weisskopf}}(E)$",
        )

    def test_channel_selection_accepts_an_iterable(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=create_compound(max_polynomial_degree=0),
            realizs=1,
            channel_indices=(index for index in (1, 0)),
        )

        self.assertEqual(simulation.channel_indices, (1, 0))
        self.assertEqual(
            tuple(
                observable.data.channel_index
                for observable in simulation.outputs.by_channel
            ),
            (1, 0),
        )


class PartialWidthOutputTests(unittest.TestCase):
    def test_default_indices_follow_compound_shape(self) -> None:
        two_channel_simulation = PartialWidthsStatisticsSimulation(
            compound=create_compound(max_polynomial_degree=0),
            realizs=1,
        )
        self.assertEqual(
            two_channel_simulation.width_indices,
            ((0, 0), (1, 0), (1, 1), (0,), (1,)),
        )

        one_channel_simulation = PartialWidthsStatisticsSimulation(
            compound=create_compound(
                max_polynomial_degree=0,
                num_free_complex_fermions=0,
            ),
            realizs=1,
        )
        self.assertEqual(
            one_channel_simulation.width_indices,
            ((0, 0), (1, 0), (0,), (1,)),
        )

    def test_invalid_indices_are_rejected_during_construction(self) -> None:
        compound = create_compound(
            max_polynomial_degree=0,
            num_free_complex_fermions=0,
        )
        invalid_cases = {
            "empty": (),
            "duplicate": ((0, 0), (0, 0)),
            "length": ((0, 0, 0),),
            "state": ((2,),),
            "channel": ((0, 1),),
        }

        for name, width_indices in invalid_cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                PartialWidthsStatisticsSimulation(
                    compound=compound,
                    realizs=1,
                    width_indices=width_indices,
                )

    def test_synthetic_accumulation_and_normalization(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=create_compound(max_polynomial_degree=0),
            realizs=2,
        )
        initial_bins = tuple(
            observable.data.bins.copy() for observable in simulation.outputs.histograms
        )

        simulation.outputs.add(np.array([[2.0, 4.0], [6.0, 8.0]]))
        simulation.outputs.add(np.array([[4.0, 8.0], [12.0, 16.0]]))
        simulation.outputs.normalize_by_average_width(simulation.realizs)

        expected_average_widths = (3.0, 9.0, 12.0, 9.0, 21.0)
        self.assertEqual(
            tuple(
                observable.metadata["average_width"]
                for observable in simulation.outputs.histograms
            ),
            expected_average_widths,
        )

        for observable, bins, average_width in zip(
            simulation.outputs.histograms,
            initial_bins,
            expected_average_widths,
            strict=True,
        ):
            self.assertEqual(observable.data.realizs, simulation.realizs)
            self.assertEqual(int(np.sum(observable.data.counts)), simulation.realizs)
            np.testing.assert_allclose(
                observable.data.bins,
                bins / average_width,
            )

    def test_zero_average_width_is_rejected_without_rescaling_bins(self) -> None:
        simulation = PartialWidthsStatisticsSimulation(
            compound=create_compound(max_polynomial_degree=0),
            realizs=1,
            width_indices=((0, 0),),
        )
        observable = simulation.outputs.histograms[0]
        initial_bins = observable.data.bins.copy()

        simulation.outputs.add(np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            simulation.outputs.normalize_by_average_width(simulation.realizs)

        self.assertEqual(observable.metadata["average_width"], 0.0)
        np.testing.assert_array_equal(observable.data.bins, initial_bins)


if __name__ == "__main__":
    unittest.main()
