# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'to_mJy',
    'inst_convolution',
]

import os
import re

import pyratbay.constants as pc
import numpy as np
import scipy.interpolate as si
from scipy.signal import convolve
from scipy.signal.windows import gaussian


ROOT = os.path.realpath(os.path.dirname(__file__)) + '/'



def to_mJy(flux, wl, units):
    """
    Convert flux spectra to mJy units.

    Parameters
    ----------
    flux: 1D float array
        SED spectrum array (see units below).
    wl: 1D float array
        SED wavelength array (microns).
    units: string
        Input units of flux:
        - 'f_freq'   (for erg s-1 cm-2 Hz-1)
        - 'f_nu'     (for for erg s-1 cm-2 cm)
        - 'f_lambda' (for erg s-1 cm-2 cm-1)

    Returns
    -------
    flux: 1D float array
        SED spectrum in mJy.
    """
    if units == 'f_freq':
        u = 10**26
    elif units == 'f_nu':
        u = 10**26 / pc.c
    elif units == 'f_lambda':
        u = 10**26 / pc.c * (wl*pc.um)**2.0
    elif 'mJy' in units:
        u = 1.0

    return flux * u


def inst_convolution(wl, spectrum, resolution, sampling_res=None, mode='same'):
    """
    Convolve a spectrum according to an instrumental resolving power

    Parameters
    ----------
    wl: 1D float array
        Spectral array (can be either wavelength or wavenumber).
    spectrum: 1D float array
        Full-resolution spectrum to be convolved.
    resolution: float
        Instrumental resolving power R = lambda/delta_lambda
        Where delta_lambda is the FHWM of the gaussian to be applied.
    sampling_red: float
        Sampling resolution of the input wl spectrum.

    Examples
    --------
    >>> import pyratbay.spectrum as ps
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> wl_min = 1.499
    >>> wl_max = 1.501
    >>> samp_resolution = 100_000
    >>> wl = ps.constant_resolution_spectrum(wl_min, wl_max, samp_resolution)
    >>> nwave = len(wl)

    >>> # A delta at wl0 ~ 1.5
    >>> spectrum = np.zeros(nwave)
    >>> spectrum[np.where(wl>1.5)[0][0]] = 1.0
    >>> wl0 = wl[np.where(wl>1.5)[0][0]]

    >>> resolution = 5_000
    >>> conv = ps.inst_convolution(wl, spectrum, resolution)

    >>> # Plot convolved line and expected FWHM
    >>> half_max = np.amax(conv)/2
    >>> hwhm = 0.5 * wl0 / resolution
    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(wl, conv, color='salmon', lw=2)
    >>> plt.plot([wl0-hwhm, wl0+hwhm], [half_max,half_max], color='xkcd:blue', lw=2)
    """
    pixel_dv = pc.c / resolution / 1e5
    n_el = int(6*pixel_dv) + 1
    kernel = gaussian(n_el, std=(pixel_dv / 2.355))
    kernel /= np.sum(kernel)

    if sampling_res is None:
        dv = pc.c/1e5 * np.ediff1d(wl) / wl[:-1]
        rv_pix = np.abs(np.mean(dv))
    else:
        rv_pix = np.abs(pc.c/1e5 / sampling_res)

    n_rv0 = int(((n_el - 1) / 2) / rv_pix)
    rv_array = np.arange(-(n_el - 1) / 2, (n_el - 1) / 2 + 1, 1)
    rv_array_mod = np.linspace(-n_rv0*rv_pix, n_rv0*rv_pix, int(2*n_rv0+1))

    csscaled = si.splrep(rv_array, kernel)
    ker_conv_pix = si.splev(rv_array_mod, csscaled, der=0)
    ker_conv_pix /= sum(ker_conv_pix)
    rconv = convolve(spectrum, ker_conv_pix, mode=mode)
    return rconv


def normalize_name(target):
    """
    Normalize target names into a 'more standard' format.
    Mainly to resolve trexolists target names.
    """
    name = re.sub(r'\s+', ' ', target)
    # It's a case issue:
    name = name.replace('KEPLER', 'Kepler')
    name = name.replace('TRES', 'TrES')
    name = name.replace('WOLF-', 'Wolf ')
    name = name.replace('HATP', 'HAT-P-')
    name = name.replace('AU-MIC', 'AU Mic')

    # Custom correction before going over prefixes
    if name.startswith('NAME-'):
        name = name[5:]
    # Prefixes
    name = name.replace('GL', 'GJ')
    prefixes = [
        'L', 'G', 'HD', 'GJ', 'LTT', 'LHS', 'HIP', 'WD',
        'LP', '2MASS', 'PSR', 'IRAS', 'TYC', 'TIC', 'PSO',
    ]
    for prefix in prefixes:
        prefix_len = len(prefix)
        if name.startswith(prefix) and not name[prefix_len].isalpha():
            name = name.replace(f'{prefix}-', f'{prefix} ')
            name = name.replace(f'{prefix}_', f'{prefix} ')
            if name[prefix_len] != ' ':
                name = f'{prefix} ' + name[prefix_len:]

    prefixes = ['CD-', 'BD-', 'BD+']
    for prefix in prefixes:
        prefix_len = len(prefix)
        dash_loc = name.find('-', prefix_len)
        if name.startswith(prefix) and dash_loc > 0:
            name = name[0:dash_loc] + ' ' + name[dash_loc+1:]
    # Main star
    if name.endswith('A') and not name[-2].isspace():
        name = name[:-1] + ' A'
    # Planet letter is in the name
    if name.lower().endswith('b') and not name[-2].isalpha():
        name = name[:-1]
    if name.lower().endswith('d') and not name[-2].isalpha():
        name = name[:-1]

    # Custom corrections
    name = name.replace('-offset', '')
    name = name.replace('-updated', '')
    name = name.replace('-copy', '')
    name = name.replace('-revised', '')
    if name.endswith('-'):
        name = name[:-1]

    if name.upper() in ['55CNC', 'RHO01-CNC', '-RHO01-CNC']:
        name = '55 Cnc'
    if name == 'WD 1856':
        name = 'WD 1856+534'
    if 'V1298' in name:
        name = 'V1298 Tau'
    return name


def is_letter(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return name[-1].islower() and name[-2] == ' '


def is_candidate(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return len(name)>=3 and name[-3] == '.' and name[-2:].isnumeric()


def get_letter(name):
    """
    Extract 'letter' identifier for a planet name.
    Valid confirmed planet names end with a lower-case letter preceded
    by a blank.  Valid planet candidate names end with a dot followed
    by two numbers.

    Examples
    --------
    >>> get_letter('TOI-741.01')
    >>> get_letter('WASP-69 b')
    """
    if is_letter(name):
        return name[-2:]
    if '.' in name:
        idx = name.rfind('.')
        return name[idx:]
    return ''


def get_host(name):
    """
    Extract host name from a given planet name.
    Valid confirmed planet names end with a lower-case letter preceded
    by a blank.  Valid planet candidate names end with a dot followed
    by two numbers.

    Examples
    --------
    >>> get_host('TOI-741.01')
    >>> get_host('WASP-69 b')
    """
    if is_letter(name):
        return name[:-2]
    if '.' in name:
        idx = name.rfind('.')
        return name[:idx]
    return ''


def select_alias(aka, catalogs, default_name=None):
    """
    Search alternative names take first one found in catalogs list.
    """
    for catalog in catalogs:
        for alias in aka:
            if alias.startswith(catalog):
                return alias
    return default_name


def invert_aliases(aliases):
    """
    Invert an {alias:name} dictionary into {name:aliases_list}
    """
    aka = {}
    for key,val in aliases.items():
        if val not in aka:
            aka[val] = []
        aka[val].append(key)
    return aka


def to_float(value):
    """
    Cast string to None or float type.
    """
    if value == 'None':
        return None
    return float(value)


def as_str(val, fmt='', if_none=None):
    """
    Format as string
    """
    if val is None or np.isnan(val):
        return if_none
    return f'{val:{fmt}}'

