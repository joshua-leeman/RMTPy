# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportUnnecessaryIsInstance=false, reportImplicitOverride=false, reportExplicitAny=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import unittest

from rmtpy.compounds import CompoundEnsemble
from rmtpy.ensembles import GaussianOrthogonalEnsemble
from rmtpy.simulations.base_plot import (
    DimensionTimeAxes,
    UnfoldedDimensionTimeAxes,
)
from rmtpy.simulations.resonance_statistics.data_factories import (
    RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
    UNFOLDED_RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
)
from rmtpy.simulations.resonance_statistics.resonance_form_factors.resonance_form_factor_plot import (
    ResonanceFormFactorsAxes,
    ResonanceFormFactorsPlot,
    UnfoldedResonanceFormFactorsAxes,
    UnfoldedResonanceFormFactorsPlot,
)
from rmtpy.simulations.spectral_statistics.spectral_form_factors.spectral_form_factors_plot import (
    FormFactorsAxes,
    FormFactorsPlot,
    UnfoldedFormFactorsAxes,
    UnfoldedFormFactorsPlot,
)
from rmtpy.simulations.statistics import LOG_D_TIME_SUPPORT, LOG_D_UNFOLDED_TIME_SUPPORT
from rmtpy.simulations.time_delay_statistics.time_delay_histograms.time_delay_histograms_plot import (
    TimeDelayHistogramAxes,
    TimeDelayHistogramPlot,
    UnfoldedTimeDelayHistogramAxes,
    UnfoldedTimeDelayHistogramPlot,
)


def build_compound(
    *,
    max_polynomial_degree: int,
    num_free_complex_fermions: int = 1,
    seed: int = 123,
) -> CompoundEnsemble:
    ensemble = GaussianOrthogonalEnsemble(
        num_majoranas=4,
        max_spectral_polynomial_degree=max_polynomial_degree,
        seed=seed,
    )
    return CompoundEnsemble(
        ensemble=ensemble,
        num_free_complex_fermions=num_free_complex_fermions,
    )


class SharedDimensionTimePlotTests(unittest.TestCase):
    def test_raw_time_plots_share_axes_and_support(self) -> None:
        for axes_cls in (
            FormFactorsAxes,
            ResonanceFormFactorsAxes,
            TimeDelayHistogramAxes,
        ):
            with self.subTest(axes_cls=axes_cls):
                self.assertTrue(issubclass(axes_cls, DimensionTimeAxes))
                axes = axes_cls()
                self.assertEqual(axes.xticks, DimensionTimeAxes().xticks)
                self.assertEqual(axes.xtick_labels, DimensionTimeAxes().xtick_labels)
                self.assertEqual(axes.xlabel, DimensionTimeAxes().xlabel)

        for plot_cls in (
            FormFactorsPlot,
            ResonanceFormFactorsPlot,
            TimeDelayHistogramPlot,
        ):
            with self.subTest(plot_cls=plot_cls):
                self.assertEqual(plot_cls.xlim, LOG_D_TIME_SUPPORT)

        self.assertEqual(
            RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
            LOG_D_TIME_SUPPORT,
        )

    def test_unfolded_time_plots_share_axes_and_support(self) -> None:
        for axes_cls in (
            UnfoldedFormFactorsAxes,
            UnfoldedResonanceFormFactorsAxes,
            UnfoldedTimeDelayHistogramAxes,
        ):
            with self.subTest(axes_cls=axes_cls):
                self.assertTrue(issubclass(axes_cls, UnfoldedDimensionTimeAxes))
                axes = axes_cls()
                reference_axes = UnfoldedDimensionTimeAxes()
                self.assertEqual(axes.xticks, reference_axes.xticks)
                self.assertEqual(axes.xtick_labels, reference_axes.xtick_labels)
                self.assertEqual(axes.xlabel, reference_axes.xlabel)

        for plot_cls in (
            UnfoldedFormFactorsPlot,
            UnfoldedResonanceFormFactorsPlot,
            UnfoldedTimeDelayHistogramPlot,
        ):
            with self.subTest(plot_cls=plot_cls):
                self.assertEqual(
                    plot_cls.xlim,
                    LOG_D_UNFOLDED_TIME_SUPPORT,
                )

        self.assertEqual(
            UNFOLDED_RESONANCE_FORM_FACTOR_LOG_D_TIME_SUPPORT,
            LOG_D_UNFOLDED_TIME_SUPPORT,
        )


if __name__ == "__main__":
    unittest.main()
