# rmtpy.plotting.cdo_evolve.__main__.py


# =======================================
# 1. Imports
# =======================================
# Standard library imports
from __future__ import annotations
from argparse import ArgumentParser

# Local application imports
from rmtpy.plotting._plot import _parse_plot_args
from rmtpy.plotting.cdo_evolve.entropy_plot import EntropyPlot
from rmtpy.plotting.cdo_evolve.observable_plot import ObservablePlot
from rmtpy.plotting.cdo_evolve.probabilities_plot import ProbabilitiesPlot
from rmtpy.plotting.cdo_evolve.purity_plot import PurityPlot


# =======================================
# 2. Main Function
# =======================================
def main() -> None:
    # Create argument parser
    parser = ArgumentParser(description="CDO Evolution Plotter")

    # Obtain file path from command line arguments
    data_path = _parse_plot_args(parser)["data_dir"]

    # Initialize observable plot
    obs_plot = ObservablePlot(data_path=data_path)

    # Plot observable plot
    obs_plot.plot()

    # Initialize observable plot
    prob_plot = ProbabilitiesPlot(data_path=data_path)

    # Plot observable plot
    prob_plot.plot()

    # Initialize observable plot
    purity_plot = PurityPlot(data_path=data_path)

    # Plot observable plot
    purity_plot.plot()

    # Initialize observable plot
    entropy_plot = EntropyPlot(data_path=data_path)

    # Plot observable plot
    entropy_plot.plot()


# If this script is run directly, execute main function
if __name__ == "__main__":
    # Run main function
    main()
