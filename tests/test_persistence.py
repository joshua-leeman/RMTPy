# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import dataclasses
import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import attrs
import numpy as np

import rmtpy.simulations as simulations
from rmtpy.compounds import CompoundEnsemble
from rmtpy.density import DensityModel
from rmtpy.ensembles import GOE, ManyBodyEnsemble
from rmtpy.simulations.base_data import Data
from rmtpy.simulations.base_plot import Plot
from rmtpy.simulations.base_simulation import RunContext, Simulation
from rmtpy.simulations.persistence import (
    MANIFEST_FILE_NAME,
    RUNTIME_DEPENDENCIES,
    PersistenceIntegrityError,
    PersistenceSchemaError,
    RunExistsError,
    collect_repository_provenance,
    compute_run_id,
)
from rmtpy.simulations.resonance_statistics import (
    ResonanceStatisticsRequest,
    ResonanceStatisticsSimulation,
    load_resonance_statistics_result,
    plot_resonance_statistics_result,
    save_resonance_statistics_result,
)
from rmtpy.simulations.resonance_statistics.complex_energy_histogram import (
    ComplexEnergyHistogramPlot,
)
from rmtpy.simulations.resonance_statistics.resonance_histogram import (
    ResonanceHistogramPlot,
)
from rmtpy.simulations.spectral_statistics import (
    SpectralStatisticsRequest,
    SpectralStatisticsSimulation,
    load_spectral_statistics_result,
    plot_spectral_statistics_result,
    save_spectral_statistics_result,
)
from rmtpy.simulations.spectral_statistics.spectral_histogram import (
    SpectralHistogramPlot,
)


def stable_software() -> dict[str, Any]:
    digest = "a" * 64
    return {
        "code_version": f"source-{digest}",
        "python": {"implementation": "CPython", "version": "3.11.0"},
        "dependencies": {name: "1.0" for name in RUNTIME_DEPENDENCIES},
        "repository": {
            "git_available": False,
            "commit": None,
            "dirty": None,
            "dirty_paths": [],
            "dirty_digest": None,
            "source_digest": digest,
        },
    }


def example_context(
    *,
    seed: int = 7,
    final_state: int = 2,
    output_request: dict[str, Any] | None = None,
) -> RunContext:
    return RunContext(
        simulation_type="example_simulation",
        result_type="ExampleResult",
        simulation_config={
            "type": "ExampleSimulation",
            "parameters": {"count": 3, "seed": seed},
        },
        output_request=(
            {"quantities": ["levels"], "modes": ["raw"]}
            if output_request is None
            else output_request
        ),
        rng={
            "policy": "numpy.random.default_rng",
            "bit_generator": "PCG64",
            "seed": seed,
            "state_policy": "capture_initial_and_final",
            "initial_state": {"state": seed},
            "final_state": {"state": final_state},
        },
        dtype={
            "configured": "float64",
            "real": "float64",
            "complex": "complex128",
        },
        execution={"calibration": {}},
    )


def small_spectral_result(*, seed: int = 11):
    return SpectralStatisticsSimulation(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=0,
            seed=seed,
        ),
        realizs=1,
        request=SpectralStatisticsRequest(
            quantities=("levels",),
            unfolding_modes=("raw",),
        ),
    ).execute()


class _FakeFigure:
    def savefig(self, path, **_kwargs) -> None:
        Path(path).write_bytes(b"complete png")


class _FailingFigure:
    def savefig(self, path, **_kwargs) -> None:
        Path(path).write_bytes(b"partial png")
        raise OSError("injected plot failure")


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, weakref_slot=False)
class _ConcretePlot(Plot):
    def plot(self, path: str | Path) -> None:
        self.build_figure()
        self.finish_plot(path)


class PersistenceTests(unittest.TestCase):
    def test_run_identity_is_canonical_sensitive_and_ignores_final_state(self) -> None:
        software = stable_software()
        first = example_context(
            output_request={"quantities": ["levels"], "modes": ["raw"]}
        )
        reordered = example_context(
            output_request={"modes": ["raw"], "quantities": ["levels"]}
        )
        self.assertEqual(
            compute_run_id(first, software=software),
            compute_run_id(reordered, software=deepcopy(software)),
        )
        self.assertEqual(
            compute_run_id(first, software=software),
            compute_run_id(
                example_context(final_state=999),
                software=software,
            ),
        )
        self.assertNotEqual(
            compute_run_id(first, software=software),
            compute_run_id(example_context(seed=8), software=software),
        )
        changed_dtype = attrs.evolve(
            first,
            dtype={
                "configured": "float32",
                "real": "float32",
                "complex": "complex64",
            },
        )
        changed_request = attrs.evolve(
            first,
            output_request={"quantities": ["levels"], "modes": ["weight"]},
        )
        self.assertNotEqual(
            compute_run_id(first, software=software),
            compute_run_id(changed_dtype, software=software),
        )
        self.assertNotEqual(
            compute_run_id(first, software=software),
            compute_run_id(changed_request, software=software),
        )
        changed_software = deepcopy(software)
        changed_software["dependencies"]["numpy"] = "2.0"
        self.assertNotEqual(
            compute_run_id(first, software=software),
            compute_run_id(first, software=changed_software),
        )
        commit = "b" * 40
        clean_software = deepcopy(software)
        clean_software["code_version"] = commit
        clean_software["repository"].update(
            git_available=True,
            commit=commit,
            dirty=False,
        )
        dirty_software = deepcopy(clean_software)
        dirty_digest = "c" * 64
        dirty_software["code_version"] = f"{commit}+dirty.{dirty_digest}"
        dirty_software["repository"].update(
            dirty=True,
            dirty_paths=["rmtpy/example.py"],
            dirty_digest=dirty_digest,
        )
        self.assertNotEqual(
            compute_run_id(first, software=clean_software),
            compute_run_id(first, software=dirty_software),
        )

    def test_manifest_is_complete_relative_and_selective(self) -> None:
        result = small_spectral_result()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_spectral_statistics_result(result, out_dir=tmp_dir)
            with (run_dir / MANIFEST_FILE_NAME).open(encoding="utf-8") as file:
                manifest = json.load(file)
            loaded = load_spectral_statistics_result(run_dir)

        self.assertEqual(
            set(manifest),
            {"schema_version", "run_id", "context", "software", "contents"},
        )
        self.assertEqual(manifest["schema_version"], simulations.SCHEMA_VERSION)
        self.assertEqual(manifest["run_id"], run_dir.name)
        self.assertEqual(
            set(manifest["context"]),
            {
                "simulation_type",
                "result_type",
                "simulation_config",
                "output_request",
                "rng",
                "dtype",
                "execution",
            },
        )
        self.assertEqual(
            set(manifest["software"]),
            {"code_version", "python", "dependencies", "repository"},
        )
        self.assertEqual(len(manifest["contents"]), 1)
        self.assertEqual(manifest["contents"][0]["logical_path"], ["raw", "levels"])
        self.assertFalse(Path(manifest["contents"][0]["relative_path"]).is_absolute())
        self.assertNotIn(str(Path(tmp_dir).resolve()), json.dumps(manifest))
        self.assertIsNotNone(loaded.raw)
        self.assertIsNone(loaded.weight)
        self.assertEqual(loaded.average_by_degree, ())
        self.assertEqual(loaded.variate_by_degree, ())

    def test_existing_run_is_never_overwritten(self) -> None:
        result = small_spectral_result(seed=12)
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_spectral_statistics_result(result, out_dir=tmp_dir)
            before = (run_dir / MANIFEST_FILE_NAME).read_bytes()
            archive = next(run_dir.rglob("*.npz"))
            archive_before = archive.read_bytes()
            result.raw.levels.counts[0] += 1
            with self.assertRaises(RunExistsError):
                save_spectral_statistics_result(result, out_dir=tmp_dir)
            self.assertEqual((run_dir / MANIFEST_FILE_NAME).read_bytes(), before)
            self.assertEqual(archive.read_bytes(), archive_before)
            self.assertEqual(
                tuple(run_dir.parent.glob(f".{run_dir.name}.partial-*")),
                (),
            )

    def test_failed_write_leaves_no_apparently_current_run(self) -> None:
        result = small_spectral_result(seed=13)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch(
                    "rmtpy.simulations.persistence.runs.write_manifest",
                    side_effect=OSError("injected failure"),
                ),
                self.assertRaisesRegex(OSError, "injected failure"),
            ):
                save_spectral_statistics_result(result, out_dir=tmp_dir)
            family_dir = Path(tmp_dir) / "spectral_statistics_simulation"
            self.assertEqual(tuple(family_dir.iterdir()), ())

    def test_schema_integrity_and_stale_content_are_rejected(self) -> None:
        result = small_spectral_result(seed=14)
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_spectral_statistics_result(result, out_dir=tmp_dir)
            manifest_path = run_dir / MANIFEST_FILE_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = simulations.SCHEMA_VERSION + 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(PersistenceSchemaError):
                load_spectral_statistics_result(run_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            stale = Path(tmp_dir) / "old_archive"
            stale.mkdir()
            (stale / "metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(PersistenceSchemaError):
                load_spectral_statistics_result(stale)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_spectral_statistics_result(result, out_dir=tmp_dir)
            (run_dir / "stale").mkdir()
            with self.assertRaises(PersistenceIntegrityError):
                load_spectral_statistics_result(run_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = save_spectral_statistics_result(result, out_dir=tmp_dir)
            archive = next(run_dir.rglob("*.npz"))
            archive.write_bytes(b"corrupted")
            with self.assertRaises(PersistenceIntegrityError):
                load_spectral_statistics_result(run_dir)

    def test_clean_and_dirty_repository_metadata_is_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repository = Path(tmp_dir) / "repository"
            package = repository / "rmtpy"
            package.mkdir(parents=True)
            source = package / "example.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
            subprocess.run(
                ("git", "config", "user.email", "tests@example.invalid"),
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ("git", "config", "user.name", "RMTPy Tests"),
                cwd=repository,
                check=True,
            )
            subprocess.run(("git", "add", "rmtpy/example.py"), cwd=repository, check=True)
            subprocess.run(
                ("git", "commit", "-qm", "initial"),
                cwd=repository,
                check=True,
            )

            clean = collect_repository_provenance(repository)
            self.assertTrue(clean["git_available"])
            self.assertFalse(clean["dirty"])
            self.assertEqual(clean["dirty_paths"], [])
            self.assertIsNone(clean["dirty_digest"])

            source.write_text("VALUE = 2\n", encoding="utf-8")
            (package / "new.py").write_text("VALUE = 3\n", encoding="utf-8")
            (repository / "unapproved.txt").write_text("ignored\n", encoding="utf-8")
            dirty = collect_repository_provenance(repository)

        self.assertTrue(dirty["dirty"])
        self.assertEqual(
            dirty["dirty_paths"],
            ["rmtpy/example.py", "rmtpy/new.py"],
        )
        self.assertEqual(len(dirty["dirty_digest"]), 64)
        self.assertTrue(
            all(not Path(path).is_absolute() for path in dirty["dirty_paths"])
        )
        self.assertNotIn(str(repository), json.dumps(dirty))

    def test_plot_output_collision_does_not_overwrite(self) -> None:
        plot = _ConcretePlot(data=Data(file_name="example"), context=example_context())
        cast(Any, plot).fig = _FakeFigure()
        cast(Any, plot).ax = object()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(type(plot.axes), "configure"),
                patch.object(type(plot.legend), "configure"),
            ):
                plot.finish_plot(tmp_dir)
                destination = Path(tmp_dir) / "example_plot.png"
                before = destination.read_bytes()
                with self.assertRaises(FileExistsError):
                    plot.finish_plot(tmp_dir)
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(tuple(Path(tmp_dir).glob("*.partial.png")), ())

    def test_failed_plot_write_leaves_no_current_or_partial_file(self) -> None:
        plot = _ConcretePlot(data=Data(file_name="example"), context=example_context())
        cast(Any, plot).fig = _FailingFigure()
        cast(Any, plot).ax = object()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(type(plot.axes), "configure"),
                patch.object(type(plot.legend), "configure"),
                self.assertRaisesRegex(OSError, "injected plot failure"),
            ):
                plot.finish_plot(tmp_dir)
            self.assertEqual(tuple(Path(tmp_dir).iterdir()), ())

    def test_loaded_spectral_and_resonance_plots_do_not_sample_or_execute(self) -> None:
        spectral_simulation = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=2,
                seed=21,
            ),
            realizs=1,
            request=SpectralStatisticsRequest(
                quantities=("levels",),
                unfolding_modes=("raw", "average"),
                degrees=(2,),
            ),
        )
        spectral = spectral_simulation.execute()
        resonance_simulation = ResonanceStatisticsSimulation(
            compound=CompoundEnsemble(
                ensemble=GOE(
                    num_majoranas=4,
                    max_spectral_polynomial_degree=2,
                    seed=22,
                )
            ),
            realizs=1,
            request=ResonanceStatisticsRequest(
                quantities=("resonances",),
                unfolding_modes=("raw", "average"),
                degrees=(2,),
            ),
        )
        resonance = resonance_simulation.execute()

        with tempfile.TemporaryDirectory() as tmp_dir:
            spectral_loaded = load_spectral_statistics_result(
                save_spectral_statistics_result(spectral, out_dir=tmp_dir)
            )
            resonance_loaded = load_resonance_statistics_result(
                save_resonance_statistics_result(resonance, out_dir=tmp_dir)
            )
            spectral_counts = spectral_loaded.raw.levels.counts.copy()
            resonance_counts = resonance_loaded.raw.resonances.counts.copy()
            with (
                patch.object(
                    DensityModel,
                    "_compute_average_coeffs",
                    side_effect=AssertionError("plotting sampled a calibration"),
                ),
                patch.object(
                    Simulation,
                    "execute",
                    side_effect=AssertionError("plotting executed a simulation"),
                ),
                patch.object(SpectralHistogramPlot, "finish_plot"),
                patch.object(ResonanceHistogramPlot, "finish_plot"),
            ):
                plot_spectral_statistics_result(
                    spectral_loaded,
                    out_dir=Path(tmp_dir) / "spectral_plots",
                    views="spectral_histogram",
                )
                plot_resonance_statistics_result(
                    resonance_loaded,
                    out_dir=Path(tmp_dir) / "resonance_plots",
                    views="resonance_histogram",
                )

        np.testing.assert_array_equal(spectral_loaded.raw.levels.counts, spectral_counts)
        np.testing.assert_array_equal(
            resonance_loaded.raw.resonances.counts,
            resonance_counts,
        )
        self.assertEqual(
            resonance_loaded.context.execution["calibration"]["timing"],
            "after_first_primary_sample",
        )

    def test_plot_structuring_does_not_mutate_run_context(self) -> None:
        result = small_spectral_result()
        before = deepcopy(result.context.simulation_config)
        plot = SpectralHistogramPlot(
            data=result.raw.levels,
            context=result.context,
        )

        ensemble = plot.structure_simulation_arg("ensemble", ManyBodyEnsemble)

        self.assertIsInstance(ensemble, ManyBodyEnsemble)
        self.assertEqual(result.context.simulation_config, before)

    def test_generator_seed_result_loads_and_plots_without_a_live_rng(self) -> None:
        source_rng = np.random.default_rng(12345)
        result = SpectralStatisticsSimulation(
            ensemble=GOE(
                num_majoranas=4,
                max_spectral_polynomial_degree=0,
                seed=source_rng,
            ),
            realizs=1,
            request=SpectralStatisticsRequest(
                quantities=("levels",),
                unfolding_modes=("raw",),
            ),
        ).execute()
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded = load_spectral_statistics_result(
                save_spectral_statistics_result(result, out_dir=tmp_dir)
            )
            with patch.object(SpectralHistogramPlot, "finish_plot"):
                plot_spectral_statistics_result(loaded, out_dir=tmp_dir)

        self.assertEqual(loaded.context.rng["seed"]["generator"], "PCG64")
        self.assertEqual(
            loaded.context.simulation_config["parameters"]["ensemble"]["parameters"][
                "seed"
            ]["generator"],
            "PCG64",
        )

    def test_loaded_complex_energy_plot_does_not_mutate_probabilities(self) -> None:
        result = ResonanceStatisticsSimulation(
            compound=CompoundEnsemble(ensemble=GOE(num_majoranas=4, seed=23)),
            realizs=1,
            request=ResonanceStatisticsRequest(
                quantities=("complex_energies",),
                unfolding_modes=("raw",),
            ),
        ).execute()
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded = load_resonance_statistics_result(
                save_resonance_statistics_result(result, out_dir=tmp_dir)
            )
            before = loaded.raw.complex_energies.histogram.copy()
            with patch.object(ComplexEnergyHistogramPlot, "finish_plot"):
                plot_resonance_statistics_result(loaded, out_dir=tmp_dir)
        np.testing.assert_array_equal(loaded.raw.complex_energies.histogram, before)
        self.assertAlmostEqual(float(np.sum(before)), 1.0)

    def test_precalibration_and_actual_execution_start_state_are_recorded(self) -> None:
        ensemble = GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=2,
            seed=24,
        )
        coefficients = ensemble.spectral_density.average_coeffs.copy()
        simulation = SpectralStatisticsSimulation(
            ensemble=ensemble,
            realizs=1,
            request=SpectralStatisticsRequest(
                quantities=("levels",),
                unfolding_modes=("average",),
                degrees=(2,),
            ),
        )
        next(ensemble.eigvals_stream(1))
        execution_start = deepcopy(ensemble.rng_state)
        result = simulation.execute()

        self.assertEqual(result.context.rng["initial_state"], execution_start)
        self.assertEqual(
            result.context.execution["calibration"]["timing"],
            "preexisting",
        )
        np.testing.assert_array_equal(
            result.context.execution["calibration"]["average_coefficients"],
            coefficients,
        )

    def test_public_contract_exposes_loaders_and_no_obsolete_orchestration(
        self,
    ) -> None:
        required = {
            "Simulation",
            "RunContext",
            "load_cdo_evolution_result",
            "load_partial_widths_statistics_result",
            "load_resonance_statistics_result",
            "load_spectral_statistics_result",
            "load_time_delay_statistics_result",
            "load_transmission_coefficients_result",
        }
        self.assertTrue(required <= set(simulations.__all__))
        self.assertTrue(all(hasattr(simulations, name) for name in simulations.__all__))
        for name in ("run", "save_metadata", "to_path"):
            self.assertFalse(hasattr(Simulation, name))
        for name in ("load", "save"):
            self.assertFalse(hasattr(Data, name))
        for data in small_spectral_result(seed=25).iterate_data():
            self.assertTrue(all(field.init for field in attrs.fields(type(data))))


if __name__ == "__main__":
    unittest.main()
