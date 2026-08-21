import unittest

import numpy as np

from rmtpy.compounds import Compound
from rmtpy.ensembles import GaussianOrthogonalEnsemble
from rmtpy.simulations.partial_widths_statistics import (
    PartialWidthsStatisticsSimulation,
)
from rmtpy.simulations.resonance_statistics import ResonanceStatisticsSimulation
from rmtpy.simulations.time_delay_statistics import TimeDelayStatisticsSimulation


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
