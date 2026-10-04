import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import attrs
import numpy as np
from matplotlib import pyplot as plt

from rmtpy.compounds import (
    CompoundEnsemble,
    PoissonCompoundEnsemble,
    SYKCompoundEnsemble,
)
from rmtpy.conversion import AttrsFields
from rmtpy.ensembles import GOE, GSE, GUE, SYK, BdGC, Poisson
from rmtpy.simulations import (
    ResonanceStatisticsSimulation,
    SpectralStatisticsSimulation,
    TimeDelayStatisticsSimulation,
    TransmissionCoefficientsSimulation,
)
from rmtpy.simulations.base_data import Data
from rmtpy.simulations.resonance_statistics.complex_energy_histogram import (
    ComplexEnergyHistogramPlot,
    UnfoldedComplexEnergyHistogramPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_form_factors import (
    FormFactorsData,
    FormFactorsPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_histogram import (
    UnfoldedSpectralHistogramPlot,
)
from rmtpy.simulations.transmission_coefficients.transmission_coefficients import (
    TransmissionCoefficientsData,
)
from rmtpy.simulations.transmission_coefficients.weisskopf_estimate import (
    WeisskopfEstimateData,
)
from tests.support import archive_fields, json_mapping


class CompoundConstructionTests(unittest.TestCase):
    def test_poisson_single_and_streaming_hamiltonians_have_physical_widths(self) -> None:
        for eigenvector_ensemble in ("GOE", "GUE", "GSE"):
            with self.subTest(eigenvector_ensemble=eigenvector_ensemble):
                ensemble = Poisson(
                    num_majoranas=6,
                    eigvec_ensemble_flag=eigenvector_ensemble,
                    seed=73,
                )
                compound = PoissonCompoundEnsemble(ensemble=ensemble)
                self.assertEqual(compound.num_channels, 3)
                self.assertEqual(compound.couplings.shape, (3,))
                self.assertEqual(
                    cast(AttrsFields, attrs.fields(type(compound)))[0].name,
                    "ensemble",
                )

                for matrix in (
                    compound.generate_effective_hamiltonian(),
                    next(compound.effective_hamiltonian_stream(1)),
                ):
                    self.assertEqual(matrix.shape, (ensemble.dimension,) * 2)
                    self.assertTrue(np.all(np.isfinite(matrix)))
                    width_matrix = 1j * (matrix - matrix.conj().T)
                    self.assertGreaterEqual(
                        float(np.min(np.linalg.eigvalsh(width_matrix))),
                        -1e-12,
                    )
                    self.assertAlmostEqual(
                        float(cast(np.complexfloating, np.trace(width_matrix)).real),
                        float(np.sum(compound.couplings**2)),
                    )

    def test_syk_construction_preserves_parity_and_channel_validation(self) -> None:
        for num_majoranas in (6, 8, 10):
            for is_even_parity in (False, True):
                with self.subTest(
                    num_majoranas=num_majoranas,
                    is_even_parity=is_even_parity,
                ):
                    ensemble = SYK(
                        q=4,
                        num_majoranas=num_majoranas,
                        is_even_parity=is_even_parity,
                        seed=17,
                    )
                    compound = SYKCompoundEnsemble(
                        ensemble=ensemble,
                        num_free_complex_fermions=0 if is_even_parity else 1,
                    )
                    self.assertEqual(
                        compound.coupling_matrix_conj.shape,
                        (compound.num_channels, ensemble.dimension),
                    )
                    self.assertTrue(
                        np.all(np.isfinite(compound.generate_effective_hamiltonian()))
                    )

        with self.assertRaisesRegex(ValueError, "share parity"):
            _ = SYKCompoundEnsemble(ensemble=SYK(q=4, num_majoranas=8))
        with self.assertRaisesRegex(ValueError, "even number of open channels"):
            _ = SYKCompoundEnsemble(
                ensemble=SYK(q=4, num_majoranas=12),
                num_free_complex_fermions=0,
            )

    def test_shared_gaussian_kernel_preserves_symmetries_and_rng_draws(self) -> None:
        for ensemble_cls in (GUE, GSE, BdGC):
            for dtype in (np.float32, np.float64):
                with self.subTest(ensemble=ensemble_cls.__name__, dtype=dtype):
                    ensemble = ensemble_cls(num_majoranas=6, dtype=dtype, seed=902)
                    matrix = ensemble.generate_matrix()
                    dimension = ensemble.dimension
                    halfway = dimension // 2
                    draws = (
                        dimension**2
                        if ensemble_cls is GUE
                        else halfway * (2 * halfway - (1 if ensemble_cls is GSE else 0))
                    )
                    reference_rng = np.random.default_rng(902)
                    _ = reference_rng.standard_normal(draws, dtype=dtype)
                    self.assertEqual(
                        ensemble.rng_state, reference_rng.bit_generator.state
                    )
                    self.assertTrue(matrix.flags.f_contiguous)
                    np.testing.assert_array_equal(matrix, matrix.conj().T)

                    eigenvalues = np.linalg.eigvalsh(matrix)
                    if ensemble_cls is GSE:
                        np.testing.assert_allclose(
                            eigenvalues[::2], eigenvalues[1::2], atol=1e-6
                        )
                    elif ensemble_cls is BdGC:
                        np.testing.assert_allclose(
                            eigenvalues, -eigenvalues[::-1], atol=1e-6
                        )


class ArchiveCompatibilityTests(unittest.TestCase):
    def test_older_unfolding_names_load_and_retain_archive_paths(self) -> None:
        compound = CompoundEnsemble(
            ensemble=GOE(num_majoranas=4, max_spectral_polynomial_degree=1, seed=37)
        )
        simulations = (
            SpectralStatisticsSimulation(ensemble=compound.ensemble, realizs=1),
            ResonanceStatisticsSimulation(compound=compound, realizs=1),
            TimeDelayStatisticsSimulation(
                compound=compound,
                energies=np.array([0.0]),
                realizs=1,
            ),
        )
        with patch("rmtpy.density.NUM_HISTOGRAM_COUNTS", 2):
            for simulation in simulations:
                with self.subTest(simulation=type(simulation).__name__):
                    simulation.execute()
                    with tempfile.TemporaryDirectory() as root:
                        destination = simulation.save(root)
                        renamed_paths: list[Path] = []
                        for data in simulation:
                            if data.metadata.get("unfolding") != "average":
                                continue
                            path = destination / data.to_path
                            fields = archive_fields(path)
                            metadata = json_mapping(cast(str, fields["metadata"].item()))
                            metadata["unfolding"] = "averaged"
                            fields["metadata"] = np.array(json.dumps(metadata))
                            old_name = data.aggregation_key.replace(
                                "_average_unfolded", "_averaged_unfolded"
                            )
                            fields["_file_name"] = np.array(old_name)
                            old_directory = destination / old_name
                            old_directory.mkdir()
                            old_path = old_directory / f"{old_name}_data.npz"
                            np.savez(old_path, allow_pickle=False, **fields)
                            path.unlink()
                            path.parent.rmdir()
                            renamed_paths.append(old_path)

                        restored = type(simulation).load(destination)
                        restored_paths = {destination / data.to_path for data in restored}
                        self.assertTrue(set(renamed_paths).issubset(restored_paths))
                        self.assertTrue(
                            all(
                                data.metadata.get("unfolding") != "averaged"
                                for data in restored
                            )
                        )

    def test_missing_single_trace_is_reconstructed_only_for_one_realization(self) -> None:
        for realizs in (1, 2):
            with self.subTest(realizs=realizs), tempfile.TemporaryDirectory() as root:
                data = FormFactorsData(dimension=2, num_times=8)
                for sample in range(realizs):
                    data.compute_moment_contributions(np.array([-0.25, 0.5 + sample]))
                data.compute_statistics()
                data.save(directory=root)
                path = Path(root) / data.to_path
                fields = archive_fields(path)
                del fields["single_realization_form_factor"]
                np.savez(path, allow_pickle=False, **fields)

                restored = Data.load(path)
                assert isinstance(restored, FormFactorsData)
                self.assertEqual(
                    restored.single_realization_form_factor_available, realizs == 1
                )
                if realizs == 1:
                    np.testing.assert_array_equal(
                        restored.single_realization_form_factor, restored.second_moment
                    )
                else:
                    context = SpectralStatisticsSimulation(
                        ensemble=GOE(num_majoranas=4), realizs=realizs
                    ).manifest
                    plot = FormFactorsPlot(data=restored, context=context)
                    with patch.object(plot, "finish_plot"):
                        plot.plot(Path("unused"))
                    try:
                        self.assertEqual(len(plot.ax.lines), 2)
                        self.assertEqual(len(plot.legend.handles), 2)
                    finally:
                        plt.close(plot.fig)

                restored.save(directory=Path(root) / "resaved")
                resaved = FormFactorsData.load(Path(root) / "resaved" / restored.to_path)
                self.assertEqual(
                    resaved.single_realization_form_factor_available, realizs == 1
                )

    def test_transmission_restores_the_archived_grid_and_rejects_disagreement(
        self,
    ) -> None:
        compound = CompoundEnsemble(ensemble=GOE(num_majoranas=4, seed=41))
        energies = np.linspace(*compound.ensemble.spectral_density.plot_range, 100)
        simulation = TransmissionCoefficientsSimulation(
            compound=compound,
            channel_indices=(0,),
            realizs=1,
            transmission_coefficient_buffers=(
                TransmissionCoefficientsData.create(energies=energies, channel_index=0),
            ),
            weisskopf_estimate_buffer=WeisskopfEstimateData.create(
                energies=energies,
                mean_level_spacings=np.ones(100),
                num_channels=compound.num_channels,
            ),
        )
        object.__setattr__(simulation, "energies", energies)
        simulation.execute()
        with tempfile.TemporaryDirectory() as root:
            destination = simulation.save(root)
            restored = TransmissionCoefficientsSimulation.load(destination)
            np.testing.assert_array_equal(restored.energies, energies)
            data = tuple(restored.transmission_coefficient_buffers)[0]
            path = destination / data.to_path
            fields = archive_fields(path)
            fields["energies"] = energies + 0.125
            np.savez(path, allow_pickle=False, **fields)
            with self.assertRaisesRegex(ValueError, "energy grids do not match"):
                _ = TransmissionCoefficientsSimulation.load(destination)


class PlotPreparationTests(unittest.TestCase):
    def test_repeated_preparation_keeps_limits_and_ticks_stable(self) -> None:
        ensemble = GOE(num_majoranas=6, seed=19)
        spectral = SpectralStatisticsSimulation(ensemble=ensemble, realizs=1)
        resonance = ResonanceStatisticsSimulation(
            compound=CompoundEnsemble(ensemble=ensemble), realizs=1
        )
        plots = (
            UnfoldedSpectralHistogramPlot(
                data=spectral.wgt_unfolded_buffers.levels, context=spectral.manifest
            ),
            ComplexEnergyHistogramPlot(
                data=resonance.raw_buffers.complex_energies, context=resonance.manifest
            ),
            UnfoldedComplexEnergyHistogramPlot(
                data=resonance.wgt_unfolded_buffers.complex_energies,
                context=resonance.manifest,
            ),
        )
        for plot in plots:
            with self.subTest(plot=type(plot).__name__):
                plot.set_derived_attributes()
                expected = (plot.xlim, plot.ylim, plot.axes.xticks, plot.axes.yticks)
                plot.set_derived_attributes()
                self.assertEqual(
                    (plot.xlim, plot.ylim, plot.axes.xticks, plot.axes.yticks), expected
                )

    def test_width_curve_aliases_accept_both_constructor_spellings(self) -> None:
        simulation = ResonanceStatisticsSimulation(
            compound=CompoundEnsemble(ensemble=GOE(num_majoranas=4)), realizs=1
        )
        plots = (
            ComplexEnergyHistogramPlot(
                data=simulation.raw_buffers.complex_energies,
                context=simulation.manifest,
                width_curve_width=2.5,
            ),
            ComplexEnergyHistogramPlot(
                data=simulation.raw_buffers.complex_energies,
                context=simulation.manifest,
                width_ENSEMBLE_AVERAGED_CURVE_WIDTH=2.5,
            ),
            ComplexEnergyHistogramPlot(
                data=simulation.raw_buffers.complex_energies,
                context=simulation.manifest,
                width_curve_width=2.5,
                width_ENSEMBLE_AVERAGED_CURVE_WIDTH=2.5,
            ),
        )
        for plot in plots:
            self.assertEqual(plot.width_curve_width, 2.5)
            self.assertEqual(plot.width_ENSEMBLE_AVERAGED_CURVE_WIDTH, 2.5)
            plot.width_curve_width = 3.0
            self.assertEqual(plot.width_ENSEMBLE_AVERAGED_CURVE_WIDTH, 3.0)
            plot.width_ENSEMBLE_AVERAGED_CURVE_WIDTH = 4.0
            self.assertEqual(plot.width_curve_width, 4.0)

        with self.assertRaisesRegex(ValueError, "Conflicting"):
            _ = ComplexEnergyHistogramPlot(
                data=simulation.raw_buffers.complex_energies,
                context=simulation.manifest,
                width_curve_width=2.5,
                width_ENSEMBLE_AVERAGED_CURVE_WIDTH=3.0,
            )

        with self.assertRaisesRegex(ValueError, "Conflicting"):
            _ = ComplexEnergyHistogramPlot(
                data=simulation.raw_buffers.complex_energies,
                context=simulation.manifest,
                width_curve_width=1.7,
                width_ENSEMBLE_AVERAGED_CURVE_WIDTH=2.5,
            )
