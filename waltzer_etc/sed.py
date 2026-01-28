# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'normalize_vega',
    'get_sed_list',
    'find_closest_sed',
    'load_sed',
    'load_sed_llmodels',
    'load_sed_phoenix',
    'blackbody',
]

import os

import astropy.units as u
import numpy as np
import pyratbay.constants as pc
import pyratbay.spectrum as ps
import synphot as syn
import synphot.units as su
from synphot.spectrum import SourceSpectrum
from synphot.models import Empirical1D

from .utils import ROOT, to_mJy

base_dir = f'{ROOT}data/models/'
sed_base_dir = f'{ROOT}data/'


def _get_sed_list(sed_type):
    """
    sed_type = 'llmodels'
    """
    if sed_type == 'llmodels':
        sed_type = 'models'
    sed_file = f'{sed_base_dir}{sed_type}/sed_files.txt'

    data = np.loadtxt(sed_file, dtype=str, delimiter=',', unpack=True)
    labels = data[0]
    teff = np.array(data[1], dtype=float)
    logg = np.array(data[2], dtype=float)
    files = data[3]
    isort = np.flip(np.argsort(teff))
    return files[isort], labels[isort], teff[isort], logg[isort]


def get_sed_types():
    """
    Get the list of SED models.

    Returns
    -------
    instruments: 1D list of strings
        JWST instruments

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> sed.get_sed_types()
    """
    return [
        'llmodels',
        'phoenix',
        #'k93models',
        'bt_settl',
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


def get_sed_list(sed_type='llmodels'):
    """
    Get all SED models for a given library.

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> files, labels, teff, logg = sed.get_sed_list('llmodels')
    >>> files, labels, teff, logg = sed.get_sed_list('bt_settl')
    >>> files, labels, teff, logg = sed.get_sed_list('phoenix')

    print(labels)
    """
    files, labels, teff, logg = _get_sed_list(sed_type)

    if sed_type == 'llmodels':
        sed_type = 'models'
    sed_folder = f'{sed_base_dir}{sed_type}'

    sed_files = np.array([os.path.join(sed_folder, file) for file in files])
    mask = [os.path.exists(file) for file in sed_files]

    return sed_files[mask], labels[mask], teff[mask], logg[mask]


def find_closest_sed(teff, logg, sed_type):
    """
    Find folder in base_dir with closest temperature to teff.

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> teff = 8000.0
    >>> file, temp = sed.find_closest_sed(8000.0, 4.5, 'llmodels')
    """
    files, labels, grid_teff, grid_logg = get_sed_list(sed_type)

    # A cost function for [teff,log_g] space
    cost = (
        np.abs(np.log10(teff/grid_teff)) +
        np.abs(logg-grid_logg) / 15.0
    )
    idx = np.argmin(cost)
    return files[idx], labels[idx], grid_teff[idx], grid_logg[idx]


def load_sed(teff=None, logg=4.5, sed_type='llmodels'):
    """
    Load the SED model that matches the T_eff and log_g the best.

    Parameters
    ----------
    teff: Float
        SED effective temperature (K).
    logg: Float
        SED log(g).

    Returns
    -------
    wl: 1D float array
        SED wavelength array (microns).
    flux: 1D float array
        SED spectrum array (mJy).

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> sed_wl, sed_flux = sed.load_sed(6050.0, sed_type='llmodels')
    >>> sed_wl, sed_flux = sed.load_sed(6050.0, sed_type='phoenix')
    >>> sed_wl, sed_flux = sed.load_sed(2080.0, sed_type='bt_settl')
    """
    # Search best match:
    file, label, teff_match, logg_match = find_closest_sed(teff, logg, sed_type=sed_type)

    if sed_type == 'llmodels':
        return load_sed_llmodels(file=file)
    if sed_type == 'bt_settl':
        return load_sed_phoenix(file=file)
    if sed_type == 'phoenix':
        return load_sed_phoenix(file=file, logg=logg_match)


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
        logg = 4.5
        file, label, teff_match, logg = find_closest_sed(teff, logg, sed_type='llmodels')

    # Load SED flux
    spectrum = np.loadtxt(file, unpack=True)
    wl, flux = spectrum[0:2]

    # Convert wavelength from angstrom to microns
    wl = wl * pc.A / pc.um

    # Convert flux to (erg s-1 cm-2 Hz-1) and then to mJy
    flux *= 4*np.pi
    flux = to_mJy(flux, wl, 'f_freq')

    return wl, flux


def load_sed_phoenix(file, logg=None):
    """
    Doc me.
    """
    if logg is None:
        sp = SourceSpectrum.from_file(file)
    else:
        flux_col = f'g{int(10*logg):02d}'
        sp = SourceSpectrum.from_file(file, flux_col=flux_col)

    wl = sp.waveset.to('micron').value
    flux = sp(sp.waveset, flux_unit=u.mJy).value

    return wl, flux


def blackbody(teff, wl=None):
    """
    Compute a Planck flux spectrum in mJy.

    Parameters
    ----------
    teff: Float
        Effective temperature (K).
    wl: 1D float array
        Wavelength array (micron).
    """
    if wl is None:
        wl = ps.constant_resolution_spectrum(0.23, 28.0, resolution=6000)

    wn = 1.0/(wl*pc.um)
    bb = ps.bbflux(wn, teff)
    bb_flux = to_mJy(bb, wl, units='f_nu')
    return bb_flux

