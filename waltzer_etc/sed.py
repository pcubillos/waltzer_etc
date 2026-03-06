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


def _get_sed_list(sed_type):
    """
    Find out which SED models are available.
    If the SED files are not there, there will be no model.
    """
    if sed_type == 'llmodels':
        sed_type = 'models'
    sed_folder = f'{sed_base_dir}{sed_type}'
    sed_file = f'{sed_base_dir}{sed_type}/sed_files.txt'

    labels = []
    teff = []
    logg = []
    files = []
    for line in open(sed_file, 'r'):
        info = [entry.strip() for entry in line.split(',')]
        # Keep only if the needed files exist:
        sed_files = info[3:]
        files_exist = np.all([
            os.path.exists(os.path.join(sed_folder, file))
            for file in sed_files
        ])
        if files_exist:
            labels.append(info[0])
            teff.append(info[1])
            logg.append(info[2])
            files.append(sed_files)

    labels = np.array(labels)
    teff = np.array(teff, dtype=float)
    logg = np.array(logg, dtype=float)
    isort = np.flip(np.argsort(teff))
    files = [files[i] for i in isort]
    return files, labels[isort], teff[isort], logg[isort]


# Run _get_sed_list() at load time to get this data once
# Then get_sed_list() will just get the info from this dictionary
SED_DATA = {
    sed_type: _get_sed_list(sed_type)
    for sed_type in get_sed_types()
}


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
    """
    if sed_type not in SED_DATA:
        msg = f'Invalid sed_type {repr(sed_type)}, select from {get_sed_types()}'
        raise ValueError(msg)

    return SED_DATA[sed_type]


def find_closest_sed(teff, logg, sed_type):
    """
    Find folder in base_dir with closest temperature to teff.

    Examples
    --------
    >>> import waltzer_etc.sed as sed
    >>> teff = 8000.0
    >>> files, label, temp, logg = sed.find_closest_sed(8000.0, 4.5, 'llmodels')
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
        return load_sed_phoenix(file=file, teff=teff_match, logg=logg_match)


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
    sed_file = f'{sed_base_dir}models/{file[0]}'
    spectrum = np.loadtxt(sed_file, unpack=True)
    wl, flux = spectrum[0:2]

    # Convert wavelength from angstrom to microns
    wl = wl * pc.A / pc.um

    # Convert flux to (erg s-1 cm-2 Hz-1) and then to mJy
    flux *= 4*np.pi
    flux = to_mJy(flux, wl, 'f_freq')

    return wl, flux


def load_sed_phoenix(file, teff=None, logg=None):
    """
    Load a PHOENIX or BT-Settl SED model.
    Not intended for the user, use load_sed() instead.
    """
    # BT-Settl models
    if logg is None:
        sed_file = f'{sed_base_dir}bt_settl/{file[0]}'
        sp = SourceSpectrum.from_file(sed_file)
    # PHOENIX models
    else:
        flux_col = f'g{int(10*logg):02d}'
        sed_file = f'{sed_base_dir}phoenix/{file[0]}'
        sp = SourceSpectrum.from_file(sed_file, flux_col=flux_col)
        if len(file) > 1:
            # Linear interpolation in temperature
            sed_file = f'{sed_base_dir}phoenix/{file[1]}'
            sp1 = SourceSpectrum.from_file(sed_file, flux_col=flux_col)
            teff0 = float(file[0][file[0].index('_')+1:file[0].index('.')])
            teff1 = float(file[1][file[1].index('_')+1:file[1].index('.')])
            sp = (sp*(teff1-teff) + sp1*(teff-teff0)) / (teff1-teff0)
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

