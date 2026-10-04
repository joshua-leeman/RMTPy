import unittest

import numpy as np

from rmtpy.simulations.histograms import Histogram, Histogram2D


class HistogramTests(unittest.TestCase):
    def test_one_dimensional_bins_are_half_open_and_count_realizations(self) -> None:
        histogram = Histogram(
            _file_name="one_dimensional_histogram",
            support=(0.0, 1.0),
            num_bins=2,
        )

        histogram.add_histogram_contribution(
            np.array([-0.1, 0.0, 0.49, 0.5, 0.99, 1.0, np.nan])
        )

        np.testing.assert_array_equal(histogram.counts, np.array([2, 2]))
        self.assertEqual(histogram.realizs, 1)

    def test_one_dimensional_density_and_probability_normalizations(self) -> None:
        histogram = Histogram(
            _file_name="one_dimensional_histogram",
            support=(0.0, 2.0),
            num_bins=2,
        )
        histogram.add_histogram_contribution(np.array([0.25, 0.75, 1.5]))

        histogram.compute_histogram()
        np.testing.assert_allclose(
            np.sum(histogram.histogram * np.diff(histogram.bins)),
            1.0,
        )

        histogram.compute_histogram_as_probabilities()
        np.testing.assert_allclose(np.sum(histogram.histogram), 1.0)

        empty_histogram = Histogram(
            _file_name="empty_histogram",
            support=(0.0, 1.0),
            num_bins=2,
        )
        empty_histogram.compute_histogram()
        np.testing.assert_array_equal(empty_histogram.histogram, np.zeros(2))

    def test_logarithmic_bins_and_constructor_arrays_are_validated(self) -> None:
        logarithmic_histogram = Histogram(
            _file_name="logarithmic_histogram",
            support=(-1.0, 1.0),
            log_base=10.0,
            num_bins=2,
        )
        np.testing.assert_allclose(logarithmic_histogram.bins, (0.1, 1.0, 10.0))

        invalid_arguments = (
            {"bins": np.array([0.0, 1.0])},
            {"bins": np.array([0.0, 0.5, 0.25])},
            {"counts": np.array([1, -1])},
            {"counts": np.array([1.0, 2.0])},
            {"histogram": np.array([0.5, np.nan])},
            {"histogram": np.array([0.5, -0.5])},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                _ = Histogram(
                    _file_name="invalid_histogram",
                    support=(0.0, 1.0),
                    num_bins=2,
                    **arguments,
                )

    def test_two_dimensional_histogram_normalizations_and_average_curve(self) -> None:
        histogram = Histogram2D(
            _file_name="two_dimensional_histogram",
            x_support=(0.0, 1.0),
            y_support=(0.0, 1.0),
            x_num_bins=2,
            y_num_bins=2,
        )
        histogram.add_histogram_contribution(
            x_data=np.array([0.1, 0.1, 0.6, 1.0]),
            y_data=np.array([0.2, 0.8, 0.8, 1.0]),
        )

        np.testing.assert_array_equal(
            histogram.counts,
            np.array([[1, 1], [0, 1]]),
        )
        self.assertEqual(histogram.realizs, 1)

        histogram.compute_histogram()
        bin_areas = np.outer(np.diff(histogram.x_bins), np.diff(histogram.y_bins))
        np.testing.assert_allclose(np.sum(histogram.histogram * bin_areas), 1.0)

        histogram.compute_histogram_probabilities()
        np.testing.assert_allclose(np.sum(histogram.histogram), 1.0)

        with np.errstate(divide="ignore"):
            x_values, average_y_values = histogram.compute_average_x_curve()
        np.testing.assert_allclose(x_values, (0.25, 0.75))
        np.testing.assert_allclose(average_y_values, (0.5, 0.75))

    def test_empty_two_dimensional_histograms_remain_well_defined(self) -> None:
        histogram = Histogram2D(
            _file_name="empty_two_dimensional_histogram",
            x_support=(0.0, 1.0),
            y_support=(0.0, 1.0),
            x_num_bins=2,
            y_num_bins=2,
        )

        histogram.compute_histogram()
        np.testing.assert_array_equal(histogram.histogram, np.zeros((2, 2)))

        histogram.compute_histogram_probabilities()
        np.testing.assert_array_equal(histogram.histogram, np.zeros((2, 2)))

        with np.errstate(divide="ignore"):
            _, average_y_values = histogram.compute_average_x_curve()
        self.assertTrue(np.all(np.isnan(average_y_values)))


if __name__ == "__main__":
    _ = unittest.main()
