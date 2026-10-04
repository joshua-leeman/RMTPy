import unittest
from pathlib import Path
from typing import cast
from unittest.mock import PropertyMock, patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rmtpy.compounds import CompoundEnsemble, PoissonCompoundEnsemble
from rmtpy.ensembles import GOE, GSE, GUE, ManyBodyEnsemble, Poisson
from rmtpy.simulations.base_plot import ENSEMBLE_AVERAGED_CURVE_WIDTH
from rmtpy.simulations.resonance_statistics import ResonanceStatisticsSimulation
from rmtpy.simulations.resonance_statistics.resonance_spacing_histogram import (
    ResonanceSpacingHistogramPlot,
    UnfoldedResonanceSpacingHistogramPlot,
)
from rmtpy.simulations.spectral_statistics import SpectralStatisticsSimulation
from rmtpy.simulations.spectral_statistics.nn_spacings_histogram import (
    SpacingsHistogramPlot,
    UnfoldedSpacingsHistogramPlot,
)
from tests.support import FloatVector

type SpacingPlot = (
    SpacingsHistogramPlot
    | UnfoldedSpacingsHistogramPlot
    | ResonanceSpacingHistogramPlot
    | UnfoldedResonanceSpacingHistogramPlot
)


def spacing_plots(ensemble: ManyBodyEnsemble) -> tuple[SpacingPlot, ...]:
    spectral = SpectralStatisticsSimulation(ensemble=ensemble, realizs=1)
    couplings = np.array([0.75, 1.0, 1.25])
    if isinstance(ensemble, Poisson):
        compound = PoissonCompoundEnsemble(ensemble=ensemble, couplings=couplings)
    else:
        compound = CompoundEnsemble(ensemble=ensemble, couplings=couplings)
    resonance = ResonanceStatisticsSimulation(
        compound=compound,
        realizs=1,
    )
    return (
        SpacingsHistogramPlot(
            data=spectral.raw_buffers.nn_spacings, context=spectral.manifest
        ),
        UnfoldedSpacingsHistogramPlot(
            data=spectral.wgt_unfolded_buffers.nn_spacings, context=spectral.manifest
        ),
        ResonanceSpacingHistogramPlot(
            data=resonance.raw_buffers.nn_spacings, context=resonance.manifest
        ),
        UnfoldedResonanceSpacingHistogramPlot(
            data=resonance.wgt_unfolded_buffers.nn_spacings, context=resonance.manifest
        ),
    )


class SpacingPlotTests(unittest.TestCase):
    def test_all_references_and_scaling_are_independent_of_ensemble(self) -> None:
        expected_labels = ("simulation", "GOE", "GUE", "GSE", "Poisson")
        expected_colors = ("#0072B2", "#009E73", "#CC79A7", "#000000")
        expected_styles = ("-", "--", "-.", ":")

        for ensemble_cls in (GOE, GUE, GSE, Poisson):
            ensemble = ensemble_cls(
                num_majoranas=6, max_spectral_polynomial_degree=0, seed=123
            )
            for plot in spacing_plots(ensemble):
                with self.subTest(
                    ensemble=ensemble_cls.__name__, plot=type(plot).__name__
                ):
                    try:
                        with patch.object(plot, "finish_plot"):
                            plot.plot(Path("unused"))

                        self.assertEqual(len(plot.ax.lines), 4)
                        self.assertEqual(plot.legend.labels, expected_labels)
                        self.assertEqual(len(plot.legend.handles), 5)
                        self.assertEqual(plot.legend.loc, "upper right")
                        self.assertEqual(plot.legend.bbox, (0.94, 0.95))
                        self.assertIsInstance(plot.legend.handles[0], Patch)
                        self.assertEqual(
                            cast(Patch, plot.legend.handles[0]).get_facecolor(),
                            to_rgba("Orange", 0.5),
                        )
                        self.assertTrue(
                            all(
                                isinstance(handle, Line2D)
                                for handle in plot.legend.handles[1:]
                            )
                        )
                        for lines in (
                            tuple(plot.ax.lines),
                            cast(tuple[Line2D, ...], plot.legend.handles[1:]),
                        ):
                            self.assertEqual(
                                tuple(line.get_color() for line in lines), expected_colors
                            )
                            self.assertEqual(
                                tuple(line.get_linestyle() for line in lines),
                                expected_styles,
                            )
                            self.assertEqual(
                                tuple(line.get_linewidth() for line in lines),
                                (ENSEMBLE_AVERAGED_CURVE_WIDTH,) * 4,
                            )
                            self.assertEqual(
                                tuple(line.get_alpha() for line in lines), (1.0,) * 4
                            )

                        self.assertEqual(
                            tuple(line.get_zorder() for line in plot.ax.lines), (2,) * 4
                        )
                        self.assertEqual(
                            tuple(line.get_label() for line in plot.ax.lines),
                            expected_labels[1:],
                        )

                        scale = cast(
                            float, plot.data.metadata.get("global_mean_spacing", 1.0)
                        )
                        spacings = (
                            cast(FloatVector, np.asarray(plot.ax.lines[0].get_xdata()))
                            / scale
                        )
                        if "global_mean_spacing" in plot.data.metadata:
                            self.assertEqual(scale, 3.0)
                            self.assertEqual(plot.xlim, (0.0, 12.0))

                        # Closed-form PDFs, including the existing GSE mean of two.
                        gse_spacings = spacings / 2
                        expected_values = (
                            np.pi / 2 * spacings * np.exp(-np.pi / 4 * spacings**2),
                            32
                            / np.pi**2
                            * spacings**2
                            * np.exp(-4 / np.pi * spacings**2),
                            2**18
                            / (3**6 * np.pi**3)
                            * gse_spacings**4
                            * np.exp(-64 / (9 * np.pi) * gse_spacings**2)
                            / 2,
                            np.exp(-spacings),
                        )
                        for line, expected in zip(
                            plot.ax.lines, expected_values, strict=True
                        ):
                            np.testing.assert_allclose(
                                cast(FloatVector, line.get_ydata()),
                                expected / scale,
                                atol=1e-14,
                            )
                    finally:
                        if hasattr(plot, "fig"):
                            plt.close(plot.fig)

    def test_legends_match_custom_histogram_and_reference_settings(self) -> None:
        ensemble = GOE(num_majoranas=6, max_spectral_polynomial_degree=0, seed=123)
        for plot in spacing_plots(ensemble):
            with self.subTest(plot=type(plot).__name__):
                plot.histogram_legend = "samples"
                plot.histogram_color = "Gold"
                plot.histogram_alpha = 0.3
                plot.surmise_width = 2.5
                plot.surmise_alpha = 0.6
                plot.surmise_zorder = 4
                try:
                    with patch.object(plot, "finish_plot"):
                        plot.plot(Path("unused"))

                    self.assertEqual(plot.legend.labels[0], "samples")
                    self.assertIsInstance(plot.legend.handles[0], Patch)
                    self.assertEqual(
                        cast(Patch, plot.legend.handles[0]).get_facecolor(),
                        to_rgba("Gold", 0.3),
                    )
                    for curve, handle in zip(
                        plot.ax.lines,
                        cast(tuple[Line2D, ...], plot.legend.handles[1:]),
                        strict=True,
                    ):
                        self.assertIsInstance(handle, Line2D)
                        self.assertEqual(curve.get_linewidth(), 2.5)
                        self.assertEqual(handle.get_linewidth(), 2.5)
                        self.assertEqual(curve.get_alpha(), 0.6)
                        self.assertEqual(handle.get_alpha(), 0.6)
                        self.assertEqual(curve.get_zorder(), 4)
                finally:
                    if hasattr(plot, "fig"):
                        plt.close(plot.fig)

    def test_references_do_not_require_an_ensemble_universality_class(self) -> None:
        ensemble = GOE(num_majoranas=6, max_spectral_polynomial_degree=0, seed=123)
        with (
            patch.object(
                ManyBodyEnsemble,
                "universality_class",
                new_callable=PropertyMock,
                return_value=None,
            ),
            patch.object(
                ManyBodyEnsemble,
                "wigner_surmise",
                side_effect=AssertionError("Ensemble-specific surmise used"),
            ),
        ):
            for plot in spacing_plots(ensemble):
                with self.subTest(plot=type(plot).__name__):
                    try:
                        with patch.object(plot, "finish_plot"):
                            plot.plot(Path("unused"))
                        self.assertEqual(len(plot.ax.lines), 4)
                        self.assertEqual(
                            plot.legend.labels,
                            ("simulation", "GOE", "GUE", "GSE", "Poisson"),
                        )
                    finally:
                        if hasattr(plot, "fig"):
                            plt.close(plot.fig)
