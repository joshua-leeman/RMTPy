# rmtpy/serialization/__init__.py

# Standard library imports
from importlib import import_module
from pathlib import Path

# Local imports
from ._converter import converter


# --------------------------
# Register Custom Converters
# --------------------------
# Dynamically import all converter modules
path: Path = Path(__file__).parent
for file in path.glob("[!_]*.py"):
    import_module(f".{file.stem}", package=__name__)
