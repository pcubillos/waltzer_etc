# Copyright (c) 2025 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'normalize_vega',
    'get_sed_list',
    'find_closest_teff',
    'load_sed_llmodels',
    #'load_sed',
]

import os
import re

import astropy.units as u
import numpy as np
import pyratbay.constants as pc
import synphot as syn
import synphot.units as su
from synphot.models import Empirical1D
from .utils import ROOT, to_mJy

base_dir = f'{ROOT}/data/models/'


def get_sed_types():
    """
    Get the list of SED models.

    Returns
    -------
    instruments: 1D list of strings
        JWST instruments
    """
    return [
        'llmodels',
        #'phoenix',
        #'k93models',
        #'bt_settl',
    ]

def load_vega():
    vega_file = f"{ROOT}/data/alpha_lyr_stis_010.fits"
    vega = syn.spectrum.SourceSpectrum.from_file(vega_file)
    return vega


# Make it available in this module
vega = load_vega()


def normalize_vega(wl, flux, v_mag):
    """
    Normalize a spectrum by the Johnson V-band magnitude
    to get flux measured at Earth.

    Parameters
    ----------
    wl: 1D darray
        Wavelength array in micron.
    flux: 1D darray
        Flux spectrum in mJy.
    v_mag: Float
        Johnson V magnitude.

    Returns
    -------
    norm_flux: 1D np.ndarrays
        Flux at Earth normalized according to V magnitude (mJy).

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> sed_wl, sed_flux = sed.load_sed_llmodels(6050.0)
    >>> norm_flux = sed.normalize_vega(sed_wl, sed_flux, v_mag=7.65)
    """
    sed = syn.spectrum.SourceSpectrum(
        Empirical1D,
        points=wl*u.micron,
        lookup_table=flux*u.mJy,
    )

    # passband
    johnson_v_file = f'{ROOT}/data/johnson_v.dat'
    johnson_v = syn.spectrum.SpectralElement.from_file(johnson_v_file)

    # Normalization
    norm_flux = v_mag * su.VEGAMAG
    norm_sed = sed.normalize(norm_flux, band=johnson_v, vegaspec=vega)
    norm_flux = norm_sed(norm_sed.waveset, flux_unit=u.mJy).value

    return norm_flux


def get_sed_list():
    """
    Get all LLMODELS SEDs
    """
    temp_pattern = re.compile(r't(\d+)g[\d\.]+', re.IGNORECASE)
    files = sorted([
        file
        for file in os.listdir(base_dir)
        if temp_pattern.search(file)
    ])
    labels = [
        f'T={int(file[1:6])}K log(g)={file[7:10]}'
        for file in files
    ]
    return files, labels


def find_closest_teff(teff):
    """
    Find folder in base_dir with closest temperature to teff.

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> teff = 8000.0
    >>> file, temp = sed.find_closest_teff(8000.0)
    """
    files, labels = get_sed_list()

    temps = [
        file[1+file.index('t'):file.index('g')]
        for file in files
    ]
    temps = np.array(temps, float)

    i = np.argmin(np.abs(temps - teff))
    return files[i], temps[i]


def load_sed_llmodels(teff=None, file=None):
    """
    Load an SED model from WALTzER grid.

    Parameters
    ----------
    teff: Float
        SED effective temperature.

    Return
    ------
    wl: 1D float array
        SED wavelength array (microns).
    flux: 1D float array
        SED spectrum array (mJy).

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> sed_wl, sed_flux = sed.load_sed_llmodels(6050.0)
    """
    # Load SED model spectrum based on temperature
    if file is None and teff is None:
        raise ValueError('one of teff or file input arguments must be set')

    if teff is not None:
        file, teff_match = find_closest_teff(teff)

    # Load SED flux
    path_file = os.path.join(base_dir, file)
    spectrum = np.loadtxt(path_file, unpack=True)
    wl, flux = spectrum[0:2]

    # Convert wavelength from angstrom to microns
    wl = wl * pc.A / pc.um

    # Convert flux to (erg s-1 cm-2 Hz-1) and then to mJy
    flux *= 4*np.pi
    flux = to_mJy(flux, wl, 'f_freq')
    # TBD: check I'm not missing a factor of A**2 or 1/A**2

    return wl, flux
