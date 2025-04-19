# rmtpy.simulations.__init__.py
"""
This __init__ file sets the environment variables to conduct HPC simulations for the RMTpy package.
It is grouped into the following sections:
    1. Imports
    2. Environment Variables
"""

# =============================
# 1. Imports
# =============================
# Standard library imports
from os import environ


# =============================
# 2. Environment Variables
# =============================
# Set number of threads for OpenBLAS
environ["OPENBLAS_NUM_THREADS"] = "1"
