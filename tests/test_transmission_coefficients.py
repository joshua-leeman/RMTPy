import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import attrs
import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import ExecutionState
from rmtpy.simulations.transmission_coefficients import (
    TransmissionCoefficientsSimulation,
    load_transmission_coefficients_simulation,
    plot_transmission_coefficients_simulation,
    run_transmission_coefficients_simulation,
)
from rmtpy.simulations.transmission_coefficients.transmission_coefficients import (
    TransmissionCoefficientsData,
    TransmissionCoefficientsPlot,
)
from rmtpy.simulations.transmission_coefficients.weisskopf_estimate import (
    WeisskopfEstimateData,
    WeisskopfEstimatePlot,
)


def build_compound(*, seed: int = 123) -> CompoundEnsemble:
    return CompoundEnsemble(
        ensemble=GOE(num_majoranas=4, seed=seed),
        couplings=np.array([0.75, 1.25]),
    )


def scattering_matrix(
    energies: np.ndarray[tuple[int], np.dtype[np.floating]],
    diagonal: tuple[complex, complex],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complexfloating]]:
    matrices = np.zeros((len(energies), 2, 2), dtype=np.complex128)
    matrices[:, 0, 0] = diagonal[0]
    matrices[:, 1, 1] = diagonal[1]
    return matrices


type TransmissionData = TransmissionCoefficientsData | WeisskopfEstimateData


def transmission_data(
    simulation: TransmissionCoefficientsSimulation,
) -> tuple[TransmissionData, ...]:
    return cast(tuple[TransmissionData, ...], tuple(simulation))


class TransmissionCoefficientsTests(unittest.TestCase):
    def test_plot_titles_preserve_channel_and_fermion_metadata(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            channel_indices=(1,),
            realizs=1,
        )
        ensemble = simulation.compound.ensemble.to_latex
        transmission_plot = TransmissionCoefficientsPlot(
            data=tuple(simulation.transmission_coefficient_buffers)[0],
            context=simulation.manifest,
        )
        weisskopf_plot = WeisskopfEstimatePlot(
            data=simulation.weisskopf_estimate_buffer,
            context=simulation.manifest,
        )
        transmission_plot.set_derived_attributes()
        weisskopf_plot.set_derived_attributes()

        self.assertEqual(
            transmission_plot.axes.title,
            "Transmission Coefficients: " + ensemble + r", $N_\text{f} = {1}$, $a = {1}$",
        )
        self.assertEqual(
            weisskopf_plot.axes.title,
            "Weisskopf Estimate: " + ensemble + r", $N_\text{f} = {1}$",
        )
        self.assertNotIn("\n", transmission_plot.axes.title)
        self.assertNotIn("\n", weisskopf_plot.axes.title)

    def test_required_channel_indices_are_normalized_and_validated(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            channel_indices=np.int64(1),
            realizs=1,
        )
        self.assertEqual(simulation.channel_indices, (1,))

        ordered = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            channel_indices=np.array([1, 0], dtype=np.int16),
            realizs=1,
        )
        self.assertEqual(ordered.channel_indices, (1, 0))

        invalid_types = (
            True,
            np.bool_(False),
            0.0,
            "0",
            (0, 1.0),
            ((0, 1),),
            np.array([[0, 1]]),
        )
        for channel_indices in invalid_types:
            with (
                self.subTest(channel_indices=channel_indices),
                self.assertRaises(TypeError),
            ):
                TransmissionCoefficientsSimulation(
                    compound=build_compound(),
                    channel_indices=channel_indices,
                    realizs=1,
                )

        invalid_values = ((), (0, 0), (-1,), (2,))
        for channel_indices in invalid_values:
            with (
                self.subTest(channel_indices=channel_indices),
                self.assertRaises(ValueError),
            ):
                TransmissionCoefficientsSimulation(
                    compound=build_compound(),
                    channel_indices=channel_indices,
                    realizs=1,
                )

        with self.assertRaises(TypeError):
            TransmissionCoefficientsSimulation(  # pyright: ignore[reportCallIssue]
                compound=build_compound(),
                realizs=1,
            )
        with self.assertRaises(TypeError):
            TransmissionCoefficientsSimulation(
                compound=build_compound(),
                channel_indices=(0,),
                realizs=1,
                num_energy_points=25,  # pyright: ignore[reportCallIssue]
            )

    def test_fixed_energy_grid_buffer_schema_and_iteration_order(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            channel_indices=(1, 0),
            realizs=2,
        )
        plot_range = simulation.compound.ensemble.spectral_density.plot_range
        expected_energies = np.linspace(
            *(1.5 * endpoint for endpoint in plot_range),
            100,
            dtype=np.float64,
        )

        np.testing.assert_array_equal(simulation.energies, expected_energies)
        self.assertFalse(simulation.energies.flags.writeable)
        self.assertEqual(
            tuple(type(data) for data in simulation),
            (
                TransmissionCoefficientsData,
                TransmissionCoefficientsData,
                WeisskopfEstimateData,
            ),
        )
        self.assertEqual(
            tuple(data._file_name for data in simulation),
            (
                "transmission_coefficients_channel_1",
                "transmission_coefficients_channel_0",
                "weisskopf_estimate",
            ),
        )

        channel_buffers = tuple(simulation.transmission_coefficient_buffers)
        self.assertEqual(
            tuple(data.metadata for data in channel_buffers),
            ({"channel_index": 1}, {"channel_index": 0}),
        )
        self.assertEqual(
            simulation.weisskopf_estimate_buffer.metadata,
            {"num_channels": 2},
        )
        self.assertEqual(simulation.weisskopf_estimate_buffer.num_channels, 2)
        self.assertEqual(simulation.weisskopf_estimate_buffer.realizs, 0)
        self.assertFalse(
            simulation.weisskopf_estimate_buffer.mean_level_spacings.flags.writeable
        )
        self.assertTrue(
            all(field.init for field in attrs.fields(TransmissionCoefficientsData))
        )
        self.assertTrue(all(field.init for field in attrs.fields(WeisskopfEstimateData)))
        self.assertFalse(hasattr(simulation, "result"))
        self.assertFalse(hasattr(simulation, "num_energy_points"))

    def test_complex_average_precedes_modulus_and_routes_selected_channels(
        self,
    ) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            channel_indices=(1, 0),
            realizs=2,
        )
        first = scattering_matrix(simulation.energies, (1.0, 0.5 + 0.5j))
        second = scattering_matrix(simulation.energies, (-1.0, -0.5 + 0.5j))

        with patch.object(
            CompoundEnsemble,
            "scattering_matrix_stream",
            return_value=iter(((first, np.empty(0)), (second, np.empty(0)))),
        ):
            returned = simulation.execute()

        self.assertIsNone(returned)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
        channel_one, channel_zero = tuple(simulation.transmission_coefficient_buffers)
        np.testing.assert_allclose(channel_one.average_scattering_diagonal, 0.5j)
        np.testing.assert_allclose(channel_one.transmission_coefficients, 0.75)
        np.testing.assert_allclose(channel_zero.average_scattering_diagonal, 0.0)
        np.testing.assert_allclose(channel_zero.transmission_coefficients, 1.0)
        np.testing.assert_allclose(
            simulation.weisskopf_estimate_buffer.average_scattering_diagonal,
            np.tile((0.0, 0.5j), (len(simulation.energies), 1)),
        )
        np.testing.assert_allclose(
            simulation.weisskopf_estimate_buffer.transmission_coefficients,
            np.tile((1.0, 0.75), (len(simulation.energies), 1)),
        )
        self.assertTrue(all(data.realizs == 2 for data in transmission_data(simulation)))

    def test_weisskopf_uses_all_channels_and_preserves_undefined_density(
        self,
    ) -> None:
        compound = build_compound()
        density = np.ones(100, dtype=np.float64)
        density[:3] = (0.0, -1.0, np.nan)
        with patch.object(
            type(compound.ensemble.spectral_density),
            "weight_pdf",
            return_value=density,
        ):
            simulation = TransmissionCoefficientsSimulation(
                compound=compound,
                channel_indices=(0,),
                realizs=1,
            )

        matrices = scattering_matrix(simulation.energies, (0.5, 0.0))
        with patch.object(
            CompoundEnsemble,
            "scattering_matrix_stream",
            return_value=iter(((matrices, np.empty(0)),)),
        ):
            simulation.execute()

        weisskopf = simulation.weisskopf_estimate_buffer
        np.testing.assert_allclose(
            weisskopf.transmission_coefficients,
            np.tile((0.75, 1.0), (len(simulation.energies), 1)),
        )
        self.assertTrue(np.all(np.isnan(weisskopf.mean_level_spacings[:3])))
        self.assertTrue(np.all(np.isnan(weisskopf.weisskopf_estimate[:3])))
        expected_spacing = 1.0 / simulation.compound.ensemble.dimension
        np.testing.assert_allclose(weisskopf.mean_level_spacings[3:], expected_spacing)
        np.testing.assert_allclose(
            weisskopf.weisskopf_estimate[3:],
            expected_spacing * 1.75 / 2.0,
        )

    def test_clipping_malformed_stream_and_one_shot_execution(self) -> None:
        clipped = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            channel_indices=(0,),
            realizs=1,
        )
        matrices = scattering_matrix(clipped.energies, (2.0, 0.0))
        with patch.object(
            CompoundEnsemble,
            "scattering_matrix_stream",
            return_value=iter(((matrices, np.empty(0)),)),
        ):
            clipped.execute()

        np.testing.assert_array_equal(
            next(
                iter(clipped.transmission_coefficient_buffers)
            ).transmission_coefficients,
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "only once"):
            clipped.execute()

        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(),
            channel_indices=(0,),
            realizs=1,
        )
        malformed = np.zeros((len(simulation.energies), 1, 1), dtype=np.complex128)
        with (
            patch.object(
                CompoundEnsemble,
                "scattering_matrix_stream",
                return_value=iter(((malformed, np.empty(0)),)),
            ),
            self.assertRaisesRegex(ValueError, "must have shape"),
        ):
            simulation.execute()

        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.save()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.plot(Path("unused"))

    def test_seeded_execution_matches_direct_stream_and_captures_rng(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(seed=314159),
            channel_indices=(1, 0),
            realizs=2,
        )
        control = build_compound(seed=314159)
        initial_rng_state = deepcopy(simulation.compound.rng_state)
        diagonal_sum = np.zeros(
            (len(simulation.energies), control.num_channels),
            dtype=np.complex128,
        )
        for matrices, _ in control.scattering_matrix_stream(
            energies=simulation.energies,
            realizs=2,
        ):
            diagonal_sum += np.diagonal(matrices, axis1=1, axis2=2)

        simulation.execute()
        expected_average = diagonal_sum / simulation.realizs
        expected_transmission = 1.0 - np.abs(expected_average) ** 2
        np.clip(expected_transmission, 0.0, 1.0, out=expected_transmission)

        self.assertEqual(simulation.compound.rng_state, control.rng_state)
        self.assertEqual(simulation.manifest.rng["initial_state"], initial_rng_state)
        self.assertEqual(simulation.manifest.rng["final_state"], control.rng_state)
        np.testing.assert_allclose(
            simulation.weisskopf_estimate_buffer.average_scattering_diagonal,
            expected_average,
        )
        np.testing.assert_allclose(
            simulation.weisskopf_estimate_buffer.transmission_coefficients,
            expected_transmission,
        )
        channel_one, channel_zero = tuple(simulation.transmission_coefficient_buffers)
        np.testing.assert_allclose(
            channel_one.transmission_coefficients,
            expected_transmission[:, 1],
        )
        np.testing.assert_allclose(
            channel_zero.transmission_coefficients,
            expected_transmission[:, 0],
        )

    def test_save_load_plot_dispatch_and_archive_validation(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(seed=77),
            channel_indices=(1, 0),
            realizs=2,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            restored_simulation = load_transmission_coefficients_simulation(
                directory=destination_directory
            )

            self.assertEqual(
                restored_simulation.execution_state,
                ExecutionState.COMPLETE,
            )
            self.assertEqual(restored_simulation.channel_indices, (1, 0))
            np.testing.assert_array_equal(
                restored_simulation.energies,
                simulation.energies,
            )
            self.assertFalse(restored_simulation.energies.flags.writeable)
            self.assertEqual(
                tuple(type(data) for data in restored_simulation),
                tuple(type(data) for data in simulation),
            )
            for restored_data, original_data in zip(
                transmission_data(restored_simulation),
                transmission_data(simulation),
                strict=True,
            ):
                self.assertEqual(restored_data.metadata, original_data.metadata)
                self.assertEqual(restored_data.realizs, original_data.realizs)
                np.testing.assert_array_equal(
                    restored_data.scattering_diagonal_sum,
                    original_data.scattering_diagonal_sum,
                )
                np.testing.assert_array_equal(
                    restored_data.average_scattering_diagonal,
                    original_data.average_scattering_diagonal,
                )
                np.testing.assert_array_equal(
                    restored_data.transmission_coefficients,
                    original_data.transmission_coefficients,
                )

            np.testing.assert_array_equal(
                restored_simulation.weisskopf_estimate_buffer.weisskopf_estimate,
                simulation.weisskopf_estimate_buffer.weisskopf_estimate,
            )

            with (
                patch.object(
                    TransmissionCoefficientsPlot,
                    "plot",
                    autospec=True,
                ) as transmission_plot,
                patch.object(
                    WeisskopfEstimatePlot,
                    "plot",
                    autospec=True,
                ) as weisskopf_plot,
            ):
                plot_transmission_coefficients_simulation(directory=destination_directory)

            self.assertEqual(transmission_plot.call_count, 2)
            weisskopf_plot.assert_called_once()
            for call in transmission_plot.call_args_list:
                self.assertIsInstance(
                    call.args[0].data,
                    TransmissionCoefficientsData,
                )
                self.assertIsNotNone(call.args[0].context)
            self.assertIsInstance(
                weisskopf_plot.call_args.args[0].data,
                WeisskopfEstimateData,
            )

            unexpected = TransmissionCoefficientsData.create(
                energies=restored_simulation.energies,
                channel_index=7,
            )
            unexpected.save(directory=destination_directory)
            with self.assertRaisesRegex(ValueError, "not part of the simulation"):
                load_transmission_coefficients_simulation(directory=destination_directory)
            (destination_directory / unexpected.to_path).unlink()

            missing_data_path = (
                destination_directory / next(iter(restored_simulation)).to_path
            )
            missing_data_path.unlink()
            with self.assertRaisesRegex(ValueError, "Saved data .* is missing"):
                load_transmission_coefficients_simulation(directory=destination_directory)
            with (
                patch.object(
                    TransmissionCoefficientsPlot,
                    "plot",
                    autospec=True,
                ) as transmission_plot,
                patch.object(
                    WeisskopfEstimatePlot,
                    "plot",
                    autospec=True,
                ) as weisskopf_plot,
                self.assertRaisesRegex(ValueError, "Saved data .* is missing"),
            ):
                restored_simulation.plot(destination_directory)

            transmission_plot.assert_not_called()
            weisskopf_plot.assert_not_called()

        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_plot_configuration_is_detached_and_does_not_advance_rng(self) -> None:
        simulation = TransmissionCoefficientsSimulation(
            compound=build_compound(seed=902),
            channel_indices=(1,),
            realizs=1,
        )
        simulation.execute()
        completed_rng_state = deepcopy(simulation.compound.rng_state)

        transmission_plot = TransmissionCoefficientsPlot(
            data=next(iter(simulation.transmission_coefficient_buffers)),
            context=simulation.manifest,
        )
        weisskopf_plot = WeisskopfEstimatePlot(
            data=simulation.weisskopf_estimate_buffer,
            context=simulation.manifest,
        )
        transmission_plot.set_derived_attributes()
        weisskopf_plot.set_derived_attributes()

        self.assertIsNot(transmission_plot.compound, simulation.compound)
        self.assertIsNot(weisskopf_plot.compound, simulation.compound)
        self.assertEqual(transmission_plot.axes.ylabel, r"$T_{1}(E)$")
        self.assertEqual(simulation.compound.rng_state, completed_rng_state)

    def test_run_helper_executes_saves_reloads_and_dispatches_plots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(
                TransmissionCoefficientsPlot,
                "plot",
                autospec=True,
            ) as transmission_plot,
            patch.object(
                WeisskopfEstimatePlot,
                "plot",
                autospec=True,
            ) as weisskopf_plot,
        ):
            simulation = run_transmission_coefficients_simulation(
                compound=build_compound(seed=311),
                channel_indices=(1,),
                realizs=1,
                directory=temporary_directory,
            )

            self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
            self.assertEqual(simulation.channel_indices, (1,))
            transmission_plot.assert_called_once()
            weisskopf_plot.assert_called_once()
            completion_time = simulation.manifest.execution["execution_time"]
            destination_directory = (
                Path(temporary_directory) / simulation.to_path / str(completion_time)
            )
            self.assertTrue((destination_directory / "manifest.json").is_file())
            self.assertTrue(
                (destination_directory / next(iter(simulation)).to_path).is_file()
            )


if __name__ == "__main__":
    unittest.main()
