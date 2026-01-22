# Copyright (c) 2025 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

from .sample_snr import *
from .snr_waltzer import *
from . import sample_snr
from . import snr_waltzer
from .version import __version__


__all__ = (
    sample_snr.__all__ +
    snr_waltzer.__all__
)

# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

