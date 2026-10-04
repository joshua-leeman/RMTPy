import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np

from rmtpy.compounds import CompoundEnsemble
from rmtpy.conversion import unwrap_json_value
from rmtpy.ensembles import GOE
from rmtpy.simulations.base_simulation import (
    ExecutionState,
    Simulation,
    SimulationManifest,
)
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation
from rmtpy.simulations.time_delay_statistics import TimeDelayStatisticsSimulation
from tests.support import json_mapping, manifest_section


def build_spectral_simulation(*, seed: int = 123) -> SpectralStatisticsSimulation:
    return SpectralStatisticsSimulation(
        ensemble=GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=0,
            seed=seed,
        ),
        realizs=1,
    )


class BaseSimulationTests(unittest.TestCase):
    def test_manifest_captures_required_configuration_rng_and_dtypes(self) -> None:
        simulation = TimeDelayStatisticsSimulation(
            compound=CompoundEnsemble(
                ensemble=GOE(num_majoranas=4, seed=123),
            ),
            energies=np.array([-0.25, 0.125]),
            realizs=2,
        )

        parameters = cast(
            dict[str, object],
            simulation.manifest.configuration["parameters"],
        )
        self.assertEqual(
            simulation.manifest.configuration["type"],
            "TimeDelayStatisticsSimulation",
        )
        self.assertEqual(parameters["realizs"], 2)
        np.testing.assert_array_equal(
            unwrap_json_value(parameters["energies"]),
            np.array([-0.25, 0.125]),
        )
        compound_parameters = cast(dict[str, object], parameters["compound"])
        self.assertEqual(compound_parameters["type"], "CompoundEnsemble")
        self.assertEqual(
            simulation.manifest.rng["initial_state"],
            simulation.compound.rng_state,
        )
        self.assertNotIn("final_state", simulation.manifest.rng)
        self.assertEqual(
            simulation.manifest.dtype,
            {
                "real": simulation.compound.ensemble.real_dtype.name,
                "complex": simulation.compound.ensemble.complex_dtype.name,
            },
        )
        self.assertEqual(
            simulation.manifest.execution,
            {"execution_state": ExecutionState.NEW},
        )

    def test_execution_is_one_shot_and_records_terminal_state(self) -> None:
        simulation = build_spectral_simulation(seed=314159)
        initial_rng_state = deepcopy(simulation.ensemble.rng_state)

        returned = simulation.execute()

        self.assertIsNone(returned)
        self.assertEqual(simulation.execution_state, ExecutionState.COMPLETE)
        self.assertEqual(simulation.manifest.rng["initial_state"], initial_rng_state)
        self.assertEqual(
            simulation.manifest.rng["final_state"],
            simulation.ensemble.rng_state,
        )
        self.assertEqual(
            simulation.manifest.execution["execution_state"],
            ExecutionState.COMPLETE,
        )
        self.assertIsInstance(
            simulation.manifest.execution["execution_time"],
            str,
        )

        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()

    def test_failed_execution_is_terminal_and_cannot_be_saved_or_plotted(
        self,
    ) -> None:
        simulation = build_spectral_simulation()

        with (
            patch.object(
                SpectralStatisticsSimulation,
                "_execute",
                side_effect=RuntimeError("expected execution failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected execution failure"),
        ):
            simulation.execute()

        self.assertEqual(simulation.execution_state, ExecutionState.FAILED)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            simulation.execute()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            _ = simulation.save()
        with self.assertRaisesRegex(RuntimeError, "only after execution"):
            simulation.plot(Path("unused"))

    def test_generic_save_and_load_restore_data_manifest_and_rng(self) -> None:
        simulation = build_spectral_simulation(seed=77)
        simulation.execute()

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            restored_simulation = Simulation.load(destination_directory)

            self.assertIsInstance(restored_simulation, SpectralStatisticsSimulation)
            restored_simulation = cast(
                SpectralStatisticsSimulation,
                restored_simulation,
            )
            self.assertEqual(
                restored_simulation.execution_state,
                ExecutionState.COMPLETE,
            )
            self.assertEqual(restored_simulation.manifest, simulation.manifest)
            self.assertEqual(
                restored_simulation.ensemble.rng_state,
                simulation.ensemble.rng_state,
            )
            self.assertEqual(
                destination_directory.relative_to(temporary_directory).parts[:2],
                ("spectral_statistics_simulation", "GOE"),
            )
            self.assertEqual(destination_directory.parent.name, "realizs_1")
            self.assertTrue((destination_directory / "manifest.json").is_file())

            for restored_data, original_data in zip(
                restored_simulation,
                simulation,
                strict=True,
            ):
                self.assertIs(type(restored_data), type(original_data))
                self.assertEqual(restored_data.metadata, original_data.metadata)
                self.assertTrue((destination_directory / restored_data.to_path).is_file())

            with self.assertRaises(FileExistsError):
                _ = simulation.save(temporary_directory)

    def test_loading_rejects_malformed_directories_and_manifest_state(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(ValueError, "malformed"),
        ):
            _ = Simulation.load(temporary_directory)

        simulation = build_spectral_simulation(seed=78)
        simulation.execute()
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            manifest_path = destination_directory / "manifest.json"
            original_manifest_text = manifest_path.read_text(encoding="utf-8")

            malformed_manifest = json_mapping(original_manifest_text)
            del manifest_section(malformed_manifest, "configuration", "parameters")[
                "realizs"
            ]
            _ = manifest_path.write_text(
                json.dumps(malformed_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing `realizs`"):
                _ = Simulation.load(destination_directory)

            _ = manifest_path.write_text(original_manifest_text, encoding="utf-8")
            malformed_manifest = json_mapping(original_manifest_text)
            manifest_section(malformed_manifest, "execution")["execution_state"] = (
                "unknown"
            )
            _ = manifest_path.write_text(
                json.dumps(malformed_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not valid"):
                _ = Simulation.load(destination_directory)

            _ = manifest_path.write_text(original_manifest_text, encoding="utf-8")
            malformed_manifest = json_mapping(original_manifest_text)
            del manifest_section(malformed_manifest, "rng")["final_state"]
            _ = manifest_path.write_text(
                json.dumps(malformed_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "final state is malformed"):
                _ = Simulation.load(destination_directory)

    def test_manifest_loading_copies_nested_values(self) -> None:
        simulation = build_spectral_simulation()
        simulation.execute()

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination_directory = simulation.save(temporary_directory)
            manifest = SimulationManifest.create_manifest_from_path(
                destination_directory / "manifest.json"
            )

        restored_parameters = cast(
            dict[str, object],
            manifest.configuration["parameters"],
        )
        restored_parameters["realizs"] = 99
        simulation_parameters = cast(
            dict[str, object],
            simulation.manifest.configuration["parameters"],
        )
        self.assertEqual(
            simulation_parameters["realizs"],
            1,
        )


if __name__ == "__main__":
    _ = unittest.main()
