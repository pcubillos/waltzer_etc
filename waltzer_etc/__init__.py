# Copyright (c) 2025 Sreejith and Patricio
# LICENSE TBD

from .version import __version__
from .snr_waltzer import *

__all__ = (
    snr_waltzer.__all__
)

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

