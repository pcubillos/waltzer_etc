# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'depth_units',
    'wl_scales',
    'flux_scales',
    'noise_choices',
    'tso_choices',
]

depth_units = [
    "none",
    "percent",
    "ppm",
]

wl_scales = {
    'Wavelength scale': {
        'linear': 'linear',
        'log': 'log',
    },
}

flux_scales = {
    'Flux scale': {
        'linear': 'linear',
        'log': 'log',
    },
}

noise_choices = {
    'variance': 'Variances',
    'snr': 'Flux S/N',
}

tso_choices = {
    'tso': 'TSO',
    'snr': 'Depth S/N',
    #'uncertainties': 'Depth uncertainties',
}
