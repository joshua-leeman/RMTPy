# rmtpy.plotting.spectral_statistics.__main__.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from argparse import ArgumentParser

# Local application imports
from rmtpy.plotting._plot import _parse_plot_args
from rmtpy.plotting.spectral_statistics.spectral_histogram import SpectralHistogram
from rmtpy.plotting.spectral_statistics.spacings_histogram import SpacingsHistogram
from rmtpy.plotting.spectral_statistics.form_factors_plot import FormFactorPlot


# =======================================
# 2. Main Function
# =======================================
def main() -> None:
    """Main function to run the spectral statistics plotting program."""
    # Create argument parser
    parser = ArgumentParser(description="Spectral Statistics Plotter")

    # Obtain file path from command line arguments
    data_path = _parse_plot_args(parser)["data_dir"]

    # Initialize spectral histogram
    spec_plot = SpectralHistogram(data_path=data_path, unfold=False)

    # Plot spectral histogram
    spec_plot.plot()

    # Initialize unfolded spectral histogram
    unf_spec_plot = SpectralHistogram(data_path=data_path, unfold=True)

    # Plot unfolded spectral histogram
    unf_spec_plot.plot()

    # Initialize spacings histogram
    spac_plot = SpacingsHistogram(data_path=data_path, unfold=False)

    # Plot spacings histogram
    spac_plot.plot()

    # Initialize unfolded spacings histogram
    unf_spac_plot = SpacingsHistogram(data_path=data_path, unfold=True)

    # Plot unfolded spacings histogram
    unf_spac_plot.plot()

    # Initialize form factor plot
    form_factor_plot = FormFactorPlot(data_path=data_path, unfold=False)

    # Plot form factor
    form_factor_plot.plot()

    # Initialize unfolded form factor plot
    unf_form_factor_plot = FormFactorPlot(data_path=data_path, unfold=True)

    # Plot unfolded form factor
    unf_form_factor_plot.plot()


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
