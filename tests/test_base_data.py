import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np

from rmtpy.simulations.base_data import (
    SAVED_CLASS_MODULE,
    SAVED_CLASS_QUALNAME,
    Data,
    graft_loaded_data,
    load_saved_data,
)
from rmtpy.simulations.histograms import Histogram, Histogram2D


def build_histogram(*, archive_name: str = "example_histogram") -> Histogram:
    histogram = Histogram(
        _file_name=archive_name,
        support=(0.0, 1.0),
        num_bins=4,
    )
    histogram.attach_metadata({"kind": "example", "indices": [0, 1]})
    histogram.add_histogram_contribution(np.array([0.1, 0.2, 0.8]))
    histogram.compute_histogram()
    return histogram


class BaseDataTests(unittest.TestCase):
    def test_data_archive_round_trip_preserves_constructor_fields(self) -> None:
        histogram = build_histogram()

        with tempfile.TemporaryDirectory() as temporary_directory:
            histogram.save(directory=temporary_directory)
            archive_path = Path(temporary_directory) / histogram.to_path

            restored_from_archive = Data.load(archive_path)
            restored_from_directory = Data.load(archive_path.parent)

        for restored_histogram in (
            restored_from_archive,
            restored_from_directory,
        ):
            with self.subTest(load_target=restored_histogram):
                self.assertIsInstance(restored_histogram, Histogram)
                restored_histogram = cast(Histogram, restored_histogram)
                self.assertEqual(restored_histogram._file_name, histogram._file_name)
                self.assertEqual(restored_histogram.metadata, histogram.metadata)
                self.assertEqual(restored_histogram.support, histogram.support)
                self.assertEqual(restored_histogram.realizs, histogram.realizs)
                np.testing.assert_array_equal(restored_histogram.bins, histogram.bins)
                np.testing.assert_array_equal(
                    restored_histogram.counts,
                    histogram.counts,
                )
                np.testing.assert_allclose(
                    restored_histogram.histogram,
                    histogram.histogram,
                )

    def test_data_loading_rejects_malformed_archives_and_wrong_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            malformed_path = directory / "malformed_data.npz"
            np.savez_compressed(malformed_path, metadata=np.array("{}"))

            with self.assertRaisesRegex(ValueError, "class is missing"):
                Data.load(malformed_path)

            missing_field_path = directory / "missing_field_data.npz"
            np.savez_compressed(
                missing_field_path,
                **{  # pyright: ignore[reportArgumentType]
                    SAVED_CLASS_MODULE: np.array(Histogram.__module__),
                    SAVED_CLASS_QUALNAME: np.array(Histogram.__qualname__),
                },
            )
            with self.assertRaisesRegex(ValueError, "missing `metadata`"):
                Data.load(missing_field_path)

            histogram_2d = Histogram2D(
                _file_name="example_histogram_2d",
                x_support=(0.0, 1.0),
                y_support=(0.0, 1.0),
                x_num_bins=2,
                y_num_bins=2,
            )
            histogram_2d.save(directory=directory)
            with self.assertRaisesRegex(TypeError, "cannot load"):
                Histogram.load(directory / histogram_2d.to_path)

            with self.assertRaisesRegex(ValueError, "malformed"):
                Data.load(directory / "missing_data.npz")

    def test_saved_data_discovery_rejects_duplicate_file_names(self) -> None:
        first_histogram = build_histogram(archive_name="first_histogram")
        second_histogram = build_histogram(archive_name="second_histogram")

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_histogram.save(directory=directory)
            second_histogram.save(directory=directory)

            loaded_data = load_saved_data(directory)
            self.assertEqual(
                set(loaded_data),
                {"first_histogram", "second_histogram"},
            )

            duplicate_directory = directory / "duplicate"
            duplicate_directory.mkdir()
            shutil.copyfile(
                directory / first_histogram.to_path,
                duplicate_directory / "duplicate_data.npz",
            )
            with self.assertRaisesRegex(ValueError, "duplicated"):
                load_saved_data(directory)

    def test_loaded_data_are_grafted_recursively_and_type_checked(self) -> None:
        expected_histogram = build_histogram(archive_name="expected_histogram")
        restored_histogram = build_histogram(archive_name="expected_histogram")
        loaded_data: dict[str, Data] = {
            restored_histogram._file_name: restored_histogram,
        }

        grafted = graft_loaded_data((expected_histogram,), loaded_data)

        self.assertEqual(grafted, (restored_histogram,))
        self.assertEqual(loaded_data, {})

        with self.assertRaisesRegex(ValueError, "is missing"):
            graft_loaded_data(expected_histogram, {})

        wrong_class = Histogram2D(
            _file_name=expected_histogram._file_name,
            x_support=(0.0, 1.0),
            y_support=(0.0, 1.0),
            x_num_bins=2,
            y_num_bins=2,
        )
        with self.assertRaisesRegex(TypeError, "has class"):
            graft_loaded_data(
                expected_histogram,
                {expected_histogram._file_name: wrong_class},
            )


if __name__ == "__main__":
    unittest.main()
