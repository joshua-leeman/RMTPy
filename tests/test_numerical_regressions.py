import unittest

import numpy as np

from rmtpy.compounds import Compound
from rmtpy.density import DensityModel
from rmtpy.ensembles import GOE, GUE, SYK, Poisson
from rmtpy.universal import porter_thomas_distribution, time_delay_pdf
from rmtpy.validators import validate_support


def deterministic_sample_stream(realizs: int):
    sample = np.linspace(-0.9, 0.9, 1024)
    for _ in range(realizs):
        yield sample


class DensityRegressionTests(unittest.TestCase):
    def test_polynomial_and_empirical_density_paths(self) -> None:
        ensemble = GOE(
            num_majoranas=4,
            max_spectral_polynomial_degree=2,
            seed=123,
        )
        polynomials = ensemble.spectral_density.compute_polynomials(np.array([0.0]))

        self.assertEqual(polynomials.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(polynomials)))

        density = DensityModel(
            dimension=1024,
            support=(-1.0, 1.0),
            sample_stream=deterministic_sample_stream,
        )
        points = np.array([-0.25, 0.25])
        average_pdf = density.average_pdf(points)
        variate_pdf = density.variate_pdf(
            points,
            sample=np.linspace(-0.9, 0.9, 1024),
        )

        self.assertTrue(np.all(np.isfinite(average_pdf)))
        self.assertTrue(np.all(np.isfinite(variate_pdf)))


class PoissonRegressionTests(unittest.TestCase):
    def test_matrix_stream_resets_its_accumulation_buffer(self) -> None:
        arguments = {
            "num_majoranas": 6,
            "max_spectral_polynomial_degree": 0,
            "seed": 123,
        }
        ensemble = Poisson(**arguments)
        reference = Poisson(**arguments)
        matrix_stream = ensemble.matrix_stream(2, use_complex_dtype=True)
        eigsys_stream = reference.eigsys_stream(2, use_complex_dtype=True)

        for _ in range(2):
            matrix = next(matrix_stream)
            expected_eigvals, _ = next(eigsys_stream)
            np.testing.assert_allclose(
                np.linalg.eigvalsh(matrix),
                expected_eigvals,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_rng_path_cdf_and_porter_thomas_wrapper(self) -> None:
        ensemble = Poisson(num_majoranas=4, seed=123)

        self.assertIs(ensemble.rng, ensemble.eigvecs_ensemble.rng)
        self.assertEqual(ensemble.cdf(np.array([ensemble.spectral_radius]))[0], 1.0)
        self.assertNotEqual(
            Poisson(num_majoranas=4, eigvecs_ensemble_flag="GOE").to_path,
            Poisson(num_majoranas=4, eigvecs_ensemble_flag="GUE").to_path,
        )

        widths = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(
            ensemble.porter_thomas_distribution(widths),
            porter_thomas_distribution(
                widths,
                dyson_index=ensemble.eigvecs_ensemble.dyson_index,
                num_channels=1,
            ),
        )


class SYKRegressionTests(unittest.TestCase):
    def test_q_is_validated_before_dependent_factories(self) -> None:
        with self.assertRaises(ValueError):
            SYK(num_majoranas=4, q=4)

    def test_parity_reaches_basis_and_density_uses_q_hermite(self) -> None:
        even = SYK(
            num_majoranas=6,
            q=4,
            is_even_parity=True,
            max_spectral_polynomial_degree=2,
        )
        odd = SYK(
            num_majoranas=6,
            q=4,
            is_even_parity=False,
            max_spectral_polynomial_degree=2,
        )

        self.assertTrue(even.majorana_fermion_basis.is_even_parity)
        self.assertFalse(odd.majorana_fermion_basis.is_even_parity)
        self.assertNotEqual(even.to_path, odd.to_path)
        self.assertEqual(
            even.spectral_density.compute_polynomials(np.array([0.0])).shape,
            (3, 1),
        )


class CompoundRegressionTests(unittest.TestCase):
    def test_complex_rotation_does_not_alias_eigenvectors(self) -> None:
        compound = Compound(
            ensemble=GUE(num_majoranas=4, seed=123),
            num_free_complex_fermions=1,
        )
        eigvecs = np.array(
            [
                [1 / np.sqrt(2), 1j / np.sqrt(2)],
                [1j / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        )
        original = eigvecs.copy()
        rotated, rotated_conj = compound.rotate_coupling_matrix_by_eigvecs(eigvecs)

        np.testing.assert_array_equal(eigvecs, original)
        np.testing.assert_allclose(rotated_conj, rotated.conj())

    def test_complex_widths_are_nonnegative_and_scattering_streams(self) -> None:
        compound = Compound(
            ensemble=GUE(num_majoranas=4, seed=9),
            num_free_complex_fermions=1,
        )

        widths = next(compound.partial_widths_stream(1))
        scattering_results = list(
            compound.scattering_matrix_stream(2, energies=np.array([0.125]))
        )

        self.assertGreaterEqual(np.min(widths), -1e-12)
        self.assertEqual(len(scattering_results), 2)

    def test_coupling_strengths_are_finite_copies(self) -> None:
        source = np.array([1.0, 2.0])
        compound = Compound(
            ensemble=GOE(num_majoranas=4),
            coupling_strengths=source,
        )
        source[0] = 9.0

        np.testing.assert_array_equal(compound.coupling_strengths, np.array([1.0, 2.0]))
        self.assertFalse(np.shares_memory(source, compound.coupling_strengths))
        self.assertFalse(compound.coupling_strengths.flags.writeable)

        for invalid_value in (np.nan, np.inf):
            self.assertRaises(
                ValueError,
                Compound,
                ensemble=GOE(num_majoranas=4),
                coupling_strengths=[1.0, invalid_value],
            )


class UniversalRegressionTests(unittest.TestCase):
    def test_integer_time_inputs_produce_floating_density(self) -> None:
        pdf = time_delay_pdf(
            np.array([1, 2]),
            num_channels=1,
            heisenberg_time=1.0,
        )

        self.assertTrue(np.issubdtype(pdf.dtype, np.floating))
        self.assertTrue(np.any(pdf > 0.0))

    def test_support_endpoints_must_be_finite(self) -> None:
        for support in ((0.0, np.inf), (0.0, np.nan)):
            self.assertRaises(ValueError, validate_support, support)


if __name__ == "__main__":
    unittest.main()
