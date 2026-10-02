import unittest
from collections.abc import Iterator

import numpy as np

from rmtpy.density import DensityModel
from rmtpy.ensembles import GOE
from rmtpy.simulations.statistics import (
    nearest_neighbor_spacings,
    scale_support,
    truncated_polynomial_degree_range,
)
from rmtpy.simulations.unfolding import (
    TruncatedPolynomialCDFFactory,
    normalize_degrees,
    unfold_values,
    unfold_widths,
)


def deterministic_sample_stream(
    realizs: int,
    *,
    use_complex_dtype: bool = False,
) -> Iterator[np.ndarray[tuple[int], np.dtype[np.floating]]]:
    del use_complex_dtype
    for _ in range(realizs):
        yield np.linspace(-0.75, 0.75, 8)


class SimulationStatisticsTests(unittest.TestCase):
    def test_spacings_supports_and_polynomial_degree_ranges(self) -> None:
        levels = np.array([3.0, 0.0, 1.0])
        np.testing.assert_array_equal(
            nearest_neighbor_spacings(levels),
            np.array([1.0, 2.0]),
        )

        degenerate_levels = np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0])
        np.testing.assert_array_equal(
            nearest_neighbor_spacings(degenerate_levels, degeneracy=2),
            np.array([1.0, 1.0, 2.0, 2.0]),
        )

        self.assertEqual(scale_support((-2.0, 3.0), scale=0.5), (-1.0, 1.5))
        self.assertEqual(tuple(truncated_polynomial_degree_range(max_degree=0)), ())
        self.assertEqual(
            tuple(truncated_polynomial_degree_range(max_degree=3)),
            (1, 2, 3),
        )

    def test_value_and_width_unfolding_apply_the_supplied_cdf(self) -> None:
        values = np.array([-0.5, 0.0, 0.5])
        unfolded_values = unfold_values(
            values,
            cdf=lambda inputs: inputs / 4.0,
            dimension=4,
        )
        np.testing.assert_allclose(unfolded_values, values)

        widths = np.array([0.25, 0.5])
        centers = np.array([-0.25, 0.25])
        unfolded_widths = unfold_widths(
            widths,
            centers=centers,
            cdf=lambda inputs: inputs,
            dimension=4,
        )
        np.testing.assert_allclose(unfolded_widths, 4.0 * widths)

    def test_truncated_polynomial_factory_orders_degrees_and_builds_cdfs(
        self,
    ) -> None:
        density = GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=2,
            seed=123,
        ).spectral_density
        factory = TruncatedPolynomialCDFFactory(
            density=density,
            degrees=(2, 1),
            density_name="spectral",
        )

        self.assertEqual(normalize_degrees((2, 1)), (1, 2))
        self.assertEqual(factory.degrees, (1, 2))
        self.assertEqual(factory.input_grid.shape, (density.num_pts,))
        self.assertEqual(factory.polynomials.shape, (3, density.num_pts))
        self.assertEqual(factory.weight.shape, (density.num_pts,))

        coefficients = np.array([1.0, 0.05, -0.01])
        interpolators = factory.interpolators_from_coeffs(coefficients)
        self.assertEqual(len(interpolators), 2)
        for interpolator in interpolators:
            self.assertTrue(np.all(np.isfinite(interpolator(np.array([-0.25, 0.25])))))

    def test_truncated_polynomial_factory_validates_density_and_coefficients(
        self,
    ) -> None:
        density_without_expansion = DensityModel(
            dimension=8,
            support=(-1.0, 1.0),
            sample_stream=deterministic_sample_stream,
        )
        empty_factory = TruncatedPolynomialCDFFactory(
            density=density_without_expansion,
            degrees=(),
        )
        self.assertEqual(empty_factory.average_interpolators(), ())
        self.assertEqual(empty_factory.input_grid.size, 0)

        with self.assertRaises(NotImplementedError):
            TruncatedPolynomialCDFFactory(
                density=density_without_expansion,
                degrees=(1,),
            )

        density = GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=2,
        ).spectral_density
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            TruncatedPolynomialCDFFactory(
                density=density,
                degrees=(3,),
            )

        factory = TruncatedPolynomialCDFFactory(
            density=density,
            degrees=(2,),
        )
        with self.assertRaisesRegex(ValueError, "shorter"):
            factory.interpolators_from_coeffs(np.array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
