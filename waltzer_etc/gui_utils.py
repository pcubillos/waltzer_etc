# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import socket
import ssl
from datetime import datetime
import re
import json

import numpy as np
import pyratbay.spectrum as ps
import pyratbay.constants as pc
import pyratbay.tools as pt
#from gen_tso import pandeia_io as jwst
from shiny import ui
from shiny.ui._card import card_body
import requests
import prompt_toolkit as ptk
from astropy.coordinates import Angle, SkyCoord
from astropy.units import hourangle, deg
from astropy.io import ascii


import waltzer_etc.sed as sed
import waltzer_etc as waltz
from waltzer_etc.utils import ROOT
from waltzer_etc.target import Target



def make_sed_catalog():
    sed_dict = {}
    sed_labels = {}
    for sed_type in sed.get_sed_types():
        sed_labels[sed_type] = {}
        sed_keys, sed_models, sed_teff, sed_logg = sed.get_sed_list(sed_type)
        for i in range(len(sed_keys)):
            spectral_type = sed_models[i].split()[0]
            key = f'{sed_type}_{spectral_type}_{sed_teff[i]:.0f}K'
            sed_labels[sed_type][key] = str(sed_models[i])
            sed_dict[key] = {
                'teff': sed_teff[i],
                'logg': sed_logg[i],
                'label': sed_models[i],
                'sed_type': sed_type,
            }
    return sed_dict, sed_labels

sed_catalog, sed_dict = make_sed_catalog()


bands_dict = {
    'gaia,g': 'Gaia mag',
    'johnson,v': 'V mag',
}

def as_str(val, fmt='', if_none=None):
    """
    Format as string
    """
    if val is None or np.isnan(val):
        return if_none
    return f'{val:{fmt}}'

def _safe_num(val, default=0.0, cast=float):
    """
    Catch empty or invalid numerical values
    Anytime a input is called as such: varname = input.varname.get()
    replace with: varname = _safe_num(input.varname.get(), default=..., cast=...)
    Only neccessary when val cant be empty string or None
    """
    if val is None or val == "":
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        return default


def get_throughput():
    det_names = ['nuv', 'vis', 'nir']
    nwave = 1000

    # A single array concatenating all three detectors
    band_wl = np.zeros(3*nwave)
    band_response = np.zeros(3*nwave)
    masks = {}

    for i,det_name in enumerate(det_names):
       det = waltz.Detector(det_name)
       wl = np.linspace(det.wl_min, det.wl_max, nwave)
       response = np.ones(nwave)
       response[-1] = 0.0

       mask = np.zeros(3*nwave, bool)
       mask[i*nwave:(i+1)*nwave] = True

       band_wl[mask] = wl
       band_response[mask] = response
       masks[det_name] = mask

    return band_wl, band_response, masks


def get_auto_sed(input):
    """
    Guess the model closest to the available options given a T_eff
    and log_g pair.
    """
    sed_type = input.sed_type()
    sed_models = sed_dict[sed_type]
    # Avoid breaking if the SED models are not there.
    if len(sed_models) == 0:
        return sed_models, ''

    try:
        t_eff = _safe_num(input.t_eff.get(), default=1400.0, cast=float)
    except ValueError:
        return sed_models, None
    logg = _safe_num(input.log_g.get(), default=4.5, cast=float)
    sed_file, label, temp, log_g = sed.find_closest_sed(t_eff, logg, sed_type)
    selected_sed = next(k for k, val in sed_models.items() if val == label)
    return sed_models, selected_sed


def planet_model_name(input):
    """
    Get the planet model name based on the transit/eclipse depth values.

    Returns
    -------
    depth_label: String
        A string representation of the depth model.
    """
    planet_model_type = input.planet_model_type.get()
    if planet_model_type == 'Input':
        return input.depth.get()
    elif planet_model_type == 'Flat':
        transit_depth = input.transit_depth.get()
        return f'Flat transit ({transit_depth:.3f}%)'
    elif planet_model_type == 'Blackbody':
        eclipse_depth = input.eclipse_depth.get()
        t_planet = input.teq_planet.get()
        return f'Blackbody({t_planet:.0f}K, rprs\u00b2={eclipse_depth:.3f}%)'


def read_depth_spectrum(input, spectra):
    """
    Parse transit/eclipse model name based on current state.
    Calculate or extract model.
    """
    model_type = input.planet_model_type.get()
    depth_label = planet_model_name(input)
    obs_geometry = input.obs_geometry.get()

    if model_type == 'Input':
        if depth_label is None:
            depth_label = 'None'
            wl = ps.constant_resolution_spectrum(0.1, 30.0, resolution=3000)
            depth = np.tile(0.0, len(wl))
        else:
            wl = spectra[obs_geometry][depth_label]['wl']
            depth = spectra[obs_geometry][depth_label]['depth']
    elif model_type == 'Flat':
        transit_depth = input.transit_depth.get() * 0.01
        wl = ps.constant_resolution_spectrum(0.1, 30.0, resolution=3000)
        nwave = len(wl)
        depth = np.tile(transit_depth, nwave)
    elif model_type == 'Blackbody':
        rprs = np.sqrt(input.eclipse_depth.get() * 0.01)
        t_planet = input.teq_planet.get()
        sed_type, sed = parse_sed(input, spectra)[0:2]
        wl, depth = jwst.blackbody_eclipse_depth(t_planet, rprs, sed_type, sed)

    return depth_label, wl, depth


def parse_obs(input):
    planet_model_type = input.planet_model_type.get()
    depth_model = None
    rprs_sq = None
    teq_planet = None
    if planet_model_type == 'Input':
        depth_model = input.depth.get()
    elif planet_model_type == 'Flat':
        rprs_sq = input.transit_depth.get()
    elif planet_model_type == 'Blackbody':
        rprs_sq = input.eclipse_depth.get()
        teq_planet = input.teq_planet.get()
    return planet_model_type, depth_model, rprs_sq, teq_planet


def make_tso_label(input, spectra):
    """Extract SED parameters"""
    name = input.target.get()
    bands = input.bands.get()
    band_names = ' + '.join(bands)

    sed_type, sed_model, norm_mag, sed_label = parse_sed(input, spectra)
    label = f'{name} / {sed_label} / {band_names}'
    return label


def parse_sed(input, spectra=None):
    """Extract SED parameters"""
    sed_type = input.sed_type()
    norm_band = 'johnson,v'
    norm_magnitude = _safe_num(input.magnitude.get(), default=10.0, cast=float)

    if sed_type in sed_dict:
        sed_model = input.sed.get()
        if sed_model not in sed_dict[sed_type]:
            return None, None, None, None

    elif sed_type == 'blackbody':
        sed_model = _safe_num(input.t_eff.get(), default=1400.0, cast=float)
        sed_model = f'bb_{sed_model:.0f}K'

    elif sed_type == 'input':
        sed_model = input.sed.get()
        if sed_model not in spectra['sed']:
            return None, None, None, None

    # Make a label
    band_name = bands_dict[norm_band].split()[0]
    band_label = f'{norm_magnitude:.2f}_{band_name}'
    sed_label = f'{sed_model}_{band_label}'

    return sed_type, sed_model, norm_magnitude, sed_label


def read_spectrum_file(file, units='none', wl_units='micron', on_fail=None):
    """
    Parameters
    ----------
    file: String
        Spectrum file to read (transit depth, eclipse depth, or stellar SED)
        This is a plain-text file with two columns (white space separater)
        First column is the wavelength, second is the depth/flux.
        Should be readable by numpy.loadtxt().
    units: String
        Units of the input spectrum.
        For depths, use one of 'none', 'percent', 'ppm'.
        For SEDs, use one of
            'mJy',
            'f_freq' (for erg s-1 cm-2 Hz-1),
            'f_nu' (for for erg s-1 cm-2 cm),
            'f_lambda' (for erg s-1 cm-2 cm-1)
    wl_units:
        Units of the input wavelength. Use one of 'micron' or 'angstrom'.
    on_fail: String
        if 'warning' raise a warning.
        if 'error' raise an error.

    Examples
    --------
    >>> import gen_tso.utils as u

    >>> file = f'{u.ROOT}data/models/WASP80b_transit.dat'
    >>> label, wl, spectra = u.read_spectrum_file(file, on_fail='warning')
    """
    # Validate units
    depth_units = [
        "none",
        "percent",
        "ppm",
    ]
    sed_units = [
        "f_freq",
        "f_nu",
        "f_lambda",
        "mJy",
    ]
    if units not in depth_units and units not in sed_units:
        msg = (
            f"The input units ({repr(units)}) must be one of {depth_units} "
            f"for depths or one of {sed_units} for SEDs"
        )
        raise ValueError(msg)

    # (try to) load the file:
    try:
        wl, spectrum = np.loadtxt(file, unpack=True)
    except ValueError as error:
        error_msg = (
            f'Error, could not load spectrum file: {repr(file)}\n{error}'
        )
        if on_fail == 'warning':
            print(error_msg)
        if on_fail == 'error':
            raise ValueError(error_msg)
        return None, None, None

    # Set the units:
    if wl_units == 'angstrom':
        wl *= pc.A / pc.um

    # Set the units:
    if units in depth_units:
        u = pt.u(units)
    else:
        if units == 'f_freq':
            u = 10**26
        elif units == 'f_nu':
            u = 10**26 / pc.c
        elif units == 'f_lambda':
            u = 10**26 / pc.c * (wl*pc.um)**2.0
        elif 'mJy' in units:
            u = 1.0

    path, label = os.path.split(file)
    if label.endswith('.dat') or label.endswith('.txt'):
        label = label[0:-4]
    return label, wl, spectrum*u


def pretty_print_target(target):
    """
    Print a target's info to HTML text.
    Must look pretty.
    """
    rplanet = as_str(target.rplanet, '.3f', '---')
    mplanet = as_str(target.mplanet, '.3f', '---')
    sma = as_str(target.sma, '.3f', '---')
    rprs = as_str(target.rprs, '.3f', '---')
    ars = as_str(target.ars, '.3f', '---')
    period = as_str(target.period, '.3f', '---')
    t_dur = as_str(target.transit_dur, '.3f', '---')
    eq_temp = as_str(target.eq_temp, '.1f', '---')

    rstar = as_str(target.rstar, '.3f', '---')
    mstar = as_str(target.mstar, '.3f', '---')
    logg = as_str(target.logg_star, '.2f', '---')
    metal = as_str(target.metal_star, '.2f', '---')
    teff = as_str(target.teff, '.1f', '---')
    v_mag = as_str(target.v_mag, '.2f', '---')

    status = 'confirmed' if target.is_confirmed else 'candidate'
    mplanet_label = 'M*sin(i)' if target.is_min_mass else 'mplanet'
    if len(target.aliases) > 0:
        aliases = f'aliases = {target.aliases}'
    else:
        aliases = ''

    color = '#15B01A' if target.is_jwst_planet else 'black'
    jwst_planet = f'<span style="color:{color}">{target.is_jwst_planet}</span>'
    color = '#15B01A' if target.is_jwst_host else 'black'
    jwst_host = f'<span style="color:{color}">{target.is_jwst_host}</span>'

    planet_info = ui.HTML(
        f'planet = {repr(target.planet)}<br>'
        f'is_jwst_planet = {jwst_planet}<br>'
        f'is_transiting = {target.is_transiting}<br>'
        f"status = '{status} planet'<br><br>"
        f"rplanet = {rplanet} r_earth<br>"
        f"{mplanet_label} = {mplanet} m_earth<br>"
        f"semi_major_axis = {sma} AU<br>"
        f"period = {period} d<br>"
        f"equilibrium_temp = {eq_temp} K<br>"
        f"transit_duration = {t_dur} h<br>"
        f"rplanet/rstar = {rprs}<br>"
        f"a/rstar = {ars}<br>"
    )

    star_info = ui.HTML(
        f'host = {repr(target.host)}<br>'
        f'is_jwst_host = {jwst_host}<br>'
        f'<br><br><br>'
        f"rstar = {rstar} r_sun<br>"
        f"mstar = {mstar} m_sun<br>"
        f"log_g = {logg}<br>"
        f"metallicity = {metal}<br>"
        f"effective_temp = {teff} K<br>"
        f"V_mag = {v_mag}<br>"
        f"RA = {target.ra:.3f} deg<br>"
        f"dec = {target.dec:.3f} deg<br>"
    )

    return planet_info, star_info, aliases


def json_target_property(name, value, format):
    """
    Create a json dictionary of a target's property (to be used on
    an overlayCatalog for ESASky).
    """
    prop = {
        'name': name,
        'value': f'{value:{format}}',
        'type': 'STRING'
    }
    return prop


def json_target(index, name, ra, dec, g_mag, teff, logg, separation):
    """
    Create a json dictionary of a target (to be used on an overlayCatalog
    for ESASky).
    """
    data = [
        json_target_property('G mag', g_mag, '.2f'),
        json_target_property('T eff', teff, '.1f'),
        json_target_property('log(g)', logg, '.2f'),
        json_target_property('Separation', separation, '.3f'),
    ]

    target = {
        'name': name,
        'id': index+1,
        'ra': f'{ra:.8f}',
        'dec': f'{dec:.8f}',
        'data': data,
    }
    return target


def esasky_js_circle(ra, dec, radius, color='#15B01A'):
    """
    Construct a JS command to draw a circle footprint for ESASky

    Parameters
    ----------
    ra: Float
        Right ascention of the center of the circle footprint (deg).
    dec: Float
        Declination of the center of the circle footprint (deg).
    radius: Float
        Radius of the circle footprint (deg).

    Returns
    -------
    footprint: Dictionary
        A dictionary with the command to draw a circle footprint
        when converted to JSON format, e.g.:
        command = json.dumps(footprint)

    For details on the ESASky JS API see:
    https://www.cosmos.esa.int/web/esdc/esasky-javascript-api

    Examples
    --------
    >>> import gen_tso.catalogs.utils as u
    >>> ra = 315.0259661
    >>> dec = -5.094857
    >>> radius = 80.0
    >>> circle = u.esasky_js_circle(ra, dec, radius)
    """
    footprint = {
        'event': 'overlayFootprints',
        'content': {
            'overlaySet': {
                'type': 'FootprintListOverlay',
                'overlayName': 'visit splitting distance',
                'cooframe': 'J2000',
                'color': color,
                'lineWidth': 5,
                'skyObjectList': [
                    {'name': 'visit splitting distance',
                     'id': 1,
                     'stcs': f'CIRCLE ICRS {ra:.8f} {dec:.8f} {radius/3600:.4f}',
                     'ra_deg': f'{ra:.8f}',
                     'dec_deg': f'{dec:.8f}',
                    }
                ]
            }
        }
    }
    return footprint

def esasky_js_catalog(query):
    """
    Construct a JS command to draw an overlayCatalog footprint for ESASky

    Parameters
    ----------
    query: List of arrays
        A list of arrays containing the names, g_mag, teff, logg,
        ra, dec, and separation of a set of targets (see Examples).

    Returns
    -------
    command: Dictionary
        A dictionary with the command to draw an overlayCatalog
        when converted to JSON format, e.g.:
        js_command = json.dumps(command)

    For details on the ESASky JS API see:
    https://www.cosmos.esa.int/web/esdc/esasky-javascript-api

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> import gen_tso.catalogs.utils as u

    >>> # Stellar sources around WASP-69:
    >>> ra_source = 315.0259661
    >>> dec_source = -5.094857
    >>> query = cat.fetch_gaia_targets(ra_source, dec_source)
    >>> circle = u.esasky_js_catalog(query)
    """
    names, g_mag, teff, logg, ra, dec, separation = query
    ntargets = len(names)
    targets = []
    for i in range(ntargets):
        target = json_target(
            i, names[i], ra[i], dec[i],
            g_mag[i], teff[i], logg[i], separation[i],
        )
        targets.append(target)

    command = {
        "event": 'overlayCatalogue',
        'content': {
            'overlaySet': {
                'type': 'SourceListOverlay',
                'overlayName': 'Nearby Gaia sources',
                'cooframe': 'J2000',
                'color': '#ee2345',
                'lineWidth': 10,
                'skyObjectList': targets,
            }
        }
    }
    return command


def custom_card(*args, body_args={}, **kwargs):
    """
    A wrapper over a Shiny card component with an explicit card_body
    (so that I can apply class_ and other customizations).
    """
    header = None
    args = list(args)
    for arg in args:
        # Only headers and footers are CardItems (right?)
        if isinstance(arg, ui.CardItem):
            header = arg
            args.remove(arg)
            break

    return ui.card(
        header,
        card_body(*args, **body_args),
        **kwargs,
    )


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


def fetch_gaia_targets(
        ra_source, dec_source, max_separation=80.0, raise_errors=True,
    ):
    """
    Search for Gaia DR3 stellar sources around a given target.

    Parameters
    ----------
    ra_source: Float
        Right ascension (deg) of target to query around.
    dec_source: Float
        Declination (deg) of target to query around.
    max_angular_distance: Float
        Maximum angular distance from target to consider (arcsec).
        Consider that the visit splitting distance for NIRSpec TA is 38"
    raise_errors: Bool
        If True and there was an error while requesting the data,
        raise the error.
        If False and there was an error, print error to screen and
        return a string identifying some known error types.

    Returns
    -------
    names: 1D string array
        Gaia DR3 stellar source names within max_separation from target
    G_mag: 1D float array
        Gaia G magnitude of found stellar sources
    teff: 1D float array
        Effective temperature (K) of found stellar source
    log_g: 1D float array
        log(g) of found stellar sources
    ra: 1D float array
        Right ascension (deg) of found stellar sources
    dec: 1D float array
        Declination (deg) of found stellar sources
    separation: 1D float array
        Angular separation (arcsec) of found stellar sources from target

    Examples
    --------
    >>> import gen_tso.catalogs as cat

    >>> # Stellar sources around WASP-69:
    >>> ra_source = 315.0259661
    >>> dec_source = -5.094857
    >>> cat.fetch_gaia_targets(ra_source, dec_source)
    """
    # Moved inside function to avoid hanging at import time
    # (when astroquery is not reachable)
    from astroquery.gaia import Gaia
    max_sep_degrees = max_separation / 3600.0

    try:
        job = Gaia.launch_job_async(
            f"""
            SELECT * \
            FROM gaiadr3.gaia_source \
            WHERE CONTAINS(\
                POINT(gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec),
                CIRCLE({ra_source}, {dec_source}, {max_sep_degrees}))=1;""",
            dump_to_file=False,
        )
    except Exception as e:
        err_text = f"Gaia astroquery request failed with {e.__class__.__name__}"
        if isinstance(e, socket.gaierror):
            print(
                f"\n{err_text}\n{str(e)}\n"
                "Likely there's no internet connection at the moment\n"
            )
            exception = 'gaierror'
        elif isinstance(e, requests.exceptions.HTTPError):
            print(
                f"\n{err_text}\n{str(e)}\n"
                f"Probably the ESA server is down at the moment\n"
            )
            exception = 'gaierror'
        elif isinstance(e, ssl.SSLError):
            print(
                f"\n{err_text}\n{str(e)}\n"
                "If you got a 'SSL: CERTIFICATE_VERIFY_FAILED' error on an "
                "OSX machine, try following the steps on this link: "
                "https://stackoverflow.com/a/42334357 which will point you to "
                "the ReadMe.rtf file in your Applications/Python 3.X folder\n"
            )
            exception = 'ssl'
        else:
            print(f"\n{err_text}\n{str(e)}")
            exception = 'other'
        if raise_errors:
            raise e
        return exception

    resp = job.get_results()
    targets = resp[~resp['teff_gspphot'].mask]

    c1 = SkyCoord(ra_source, dec_source, unit='deg', frame='icrs')
    separation = []
    for i,target in enumerate(targets):
        c2 = SkyCoord(target['ra'], target['dec'], unit='deg', frame='icrs')
        sep = c1.separation(c2).to('arcsec').value
        separation.append(sep)

    sep_isort = np.argsort(separation)
    for i,idx in enumerate(sep_isort):
        target = targets[idx]

    return (
        targets['designation'].data.data[sep_isort],
        targets['phot_g_mean_mag'].data.data[sep_isort],
        targets['teff_gspphot'].data.data[sep_isort],
        targets['logg_gspphot'].data.data[sep_isort],
        targets['ra'].data.data[sep_isort],
        targets['dec'].data.data[sep_isort],
        np.array(separation)[sep_isort],
    )


def load_targets(database='nea_data.txt', is_confirmed=np.nan):
    """
    Unpack star and planet properties from plain text file.

    Parameters
    ----------
    database: String
        nea_data.txt or tess_data.txt

    Returns
    -------
    targets: List of Target

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> nea_data = cat.load_nea_targets_table()
    """
    with open(f'{ROOT}data/{database}', 'r') as f:
        lines = f.readlines()

    lines = [
        line for line in lines
        if not line.strip().startswith('#')
    ]
    targets = []
    for line in lines:
        if line.startswith('>'):
            name_len = line.find(':')
            host = line[1:name_len]
            star_vals = np.array(line[name_len+1:].split(), float)
            ra, dec, v_mag, rstar, mstar, teff, logg, metal = star_vals
        elif line.startswith(' '):
            name_len = line.find(':')
            planet = line[1:name_len].strip()
            planet_vals = np.array(line[name_len+1:].split(), float)
            t_dur, rplanet, mplanet, sma, period, teq, min_mass = planet_vals

            target = Target(
                host=host,
                mstar=mstar, rstar=rstar, teff=teff, logg_star=logg,
                metal_star=metal,
                v_mag=v_mag, ra=ra, dec=dec,
                planet=planet,
                mplanet=mplanet, rplanet=rplanet,
                period=period, sma=sma, transit_dur=t_dur,
                is_confirmed=is_confirmed,
                is_min_mass=bool(min_mass),
            )
            targets.append(target)

    return targets


def _add_planet_info(observations):
    """
    Add planet letter info to a list of observations
    and other corrections
    """
    planets_file = f'{ROOT}data/planets_per_program.json'
    with open(planets_file, "r") as f:
        planet_data = json.load(f)

    known_obs = []
    for obs in observations:
        pid = obs['pid']
        obs_id = obs['observation']
        key = f'{pid}_{obs_id}'
        known_obs.append(key)
        if key in planet_data:
            for var, value in planet_data[key].items():
                obs[var] = value

    for key,obs in planet_data.items():
        if key not in known_obs and 'missing' in obs:
            obs.pop('missing')
            date_format = "%Y-%m-%d %H:%M:%S"
            val = obs['date_start']
            if isinstance(val, str):
                obs['date_start'] = datetime.strptime(val, date_format)
            val = obs['date_end']
            if isinstance(val, str):
                obs['date_end'] = datetime.strptime(val, date_format)
            observations.append(obs)

    return observations


def _group_by_target(observations):
    """
    Group observations by host, using RA and dec to detect aliases
    for a same object
    """
    ra = [Angle(obs['ra'], unit=hourangle).deg for obs in observations]
    dec = [Angle(obs['dec'], unit=deg).deg for obs in observations]
    coords = SkyCoord(ra, dec, unit='deg', frame='icrs')

    nobs = len(observations)
    taken = np.zeros(nobs, bool)
    group_indices = []
    for i in range(nobs):
        if taken[i]:
            continue
        seps = coords[i].separation(coords).to('arcsec').value
        indices = np.where(seps < 50)[0]
        taken[indices] = True
        group_indices.append(indices)

    grouped_data = []
    for i,indices in enumerate(group_indices):
        target = [observations[j] for j in indices]
        grouped_data.append(target)

    return grouped_data


def load_trexolists(grouped=False, trexo_file=None):
    """
    Extract the JWST programs' data from a trexolists.csv file.
    Note that trexolists know targets by their host star, not by
    individual planets in a given system.

    Parameters
    ----------
    grouped: Bool
        - If False, return a single 1D list with observations.
        - If True, return a nested list of the observations per target.
    trexo_file: String
        If None, extract data from default Gen TSO location.
        Otherwise, a path to a trexolists.csv file.

    Returns
    -------
    observations: 1D or 2D list of dictionaries
        A list of all JWST observations, where each item is a dictionary
        containing the observation's details.
        If grouped is True, the output is a nested list, where the
        observations are grouped per target (host).

    """
    if trexo_file is None:
        trexo_file = f'{ROOT}data/trexolists.csv'

    trexolist_data = ascii.read(
        trexo_file,
        format='csv', guess=False, fast_reader=False, comment='#',
    )

    nirspec_filter = {
        'G395H': 'F290LP',
        'G395M': 'F290LP',
        'G235H': 'F170LP',
        'G235M': 'F170LP',
        'G140H': 'F100LP',
        'G140M': 'F100LP',
        'PRISM': 'CLEAR',
    }
    instrument = {
        'BOTS': 'NIRSPEC',
        'SOSS': 'NIRISS',
        'GTS': 'NIRCAM',
        'LRS': 'MIRI',
        'MRS': 'MIRI',
        'F1500W': 'MIRI',
        'F1280W': 'MIRI',
    }

    observations = []
    for i,data in enumerate(trexolist_data):
        obs = {}
        obs['category'] = str(data['ProposalCategory'])
        obs['pi'] = str(data['LastName'])
        obs['pid'] = str(data['ProposalID'])
        obs['cycle'] = str(data['Cycle'])
        obs['proprietary_period'] = int(data['ProprietaryPeriod'])

        target = str(data['hostname_nn'])
        obs['target'] = normalize_name(target)
        obs['target_in_program'] = target
        obs['planets'] = data['letter_nn'].split('+')
        obs['event'] = data['Event'].lower().replace('phasec', 'phase curve')

        obs['observation'] = str(data['Observation'])
        obs['visit'] = '1'
        obs['status'] = str(data['Status'])
        coordinates = data['EquatorialCoordinates'].split()
        obs['ra'] = ':'.join(coordinates[0:3])
        obs['dec'] = ':'.join(coordinates[3:6])

        mode = str(data['ObservingMode'])
        disperser = obs['disperser'] = str(data['GratingGrism'])
        inst = obs['instrument'] = instrument[mode]
        if mode == 'SOSS':
            disperser = 'None'
            filter = 'CLEAR'
        elif mode == 'LRS':
            disperser = 'None'
            filter = 'None'
        elif mode == 'MRS':
            disperser = 'unknown'
            filter = 'None'
        elif inst == 'MIRI':
            # disperser will be fixed below by _add_planet_info()
            disperser = 'None'
            filter = mode
            mode = 'Imaging TS'
        elif mode == 'GTS':
            mode = 'GRISMR TS'
            disperser, filter = disperser.split('+')
            if '_' in data['Subarray']:
                disperser = f'DHS0,{disperser}'
                # hard-coded, known up to Cycle4:
                # will be fixed below by _add_planet_info()
                filter = f'F150W2,{filter}'
        elif inst == 'NIRSPEC':
            filter = nirspec_filter[disperser]
        obs['mode'] = mode
        obs['disperser'] = disperser
        obs['filter'] = filter

        obs['subarray'] = str(data['Subarray'])
        obs['readout'] = str(data['ReadoutPattern'])
        obs['groups'] = int(data['Groups'])

        window = str(data['PlanWindow'])
        if window == 'X':
            obs['plan_window'] = None
        elif '(' in window:
            window = window[0:window.index('(')]
            w_start, w_end = window.split('-')
            start = datetime.strptime(w_start.strip(), '%b %d, %Y')
            end = datetime.strptime(w_end.strip(), '%b %d, %Y')
            obs['plan_window'] = f"{start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}"
        else:
            obs['plan_window'] = window

        obs['duration'] = float(data['Hours'])

        date = data['StartTime']
        if date == 'X':
            obs['date_start'] = None
        else:
            obs['date_start'] = datetime.strptime(date, '%b %d, %Y %H:%M:%S')
        date = data['EndTime']
        if date == 'X':
            obs['date_end'] = None
        else:
            obs['date_end'] = datetime.strptime(date, '%b %d, %Y %H:%M:%S')

        observations.append(obs)

    observations = _add_planet_info(observations)

    if grouped:
        return _group_by_target(observations)
    return observations


def is_letter(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return name[-1].islower() and name[-2] == ' '


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


def parse(name, style):
    """
    Parse a planet name as a planet (no change) or as a star,
    in which case it will identify from confirmed or candidate
    format name.

    Parameters
    ----------
    style: String
        Select from 'planet' or 'host'.

    Returns
    -------
    name: String
        Parsed name.

    Examples
    --------
    >>> from gen_tso.catalogs.catalogs import parse
    >>> parse('WASP-80 b', 'planet')
    WASP-80 b'
    >>> parse('WASP-80 b', 'host')
    WASP-80'
    >>> parse('TOI-316.01', 'host')
    'TOI-316'
    """
    if style == 'planet':
        return name
    elif style == 'host':
        if is_letter(name):
            return name[:-2]
        end = name.rindex('.')
        return name[:end]


def load_aliases(style='planet', aliases_file=None):
    """
    Load file with known aliases of NEA targets.

    Parameters
    ----------
    style: String
        Select from 'planet', 'host', or 'system'.

    Returns
    -------
    aliases: Dictionary
        Dictionary of aliases to-from NASA Exoplanet Archive name.
        See below for examples depending on the 'style' argument.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>>
    >>> # From alias planet name to NEA name:
    >>> aliases = cat.load_aliases('planet')
    >>> aliases['CD-38 2551 b']
    'WASP-63 b'
    >>>
    >>> # From alias host name to NEA name:
    >>> aliases = cat.load_aliases('host')
    >>> aliases['CD-38 2551']
    'WASP-63'
    >>>
    >>> # As stellar system with all host and planet aliases:
    >>> aliases = cat.load_aliases('system')
    >>> aliases['WASP-63']
    {'host': 'WASP-63',
     'planets': ['WASP-63 b'],
     'host_aliases': array(['CD-38 2551', 'TOI-483', 'WASP-63'], dtype='<U10'),
     'planet_aliases': {'TOI-483.01': 'WASP-63 b',
      'CD-38 2551 b': 'WASP-63 b',
      'WASP-63 b': 'WASP-63 b'}}
    """
    if style not in ['planet', 'host', 'system']:
        raise ValueError(
            "Invalid alias style, select from: 'planet', 'host', or 'system'"
        )

    if aliases_file is None:
        aliases_file = f'{ROOT}data/target_aliases.txt'

    with open(aliases_file, 'r') as f:
        lines = f.readlines()

    if style != 'system':
        aliases = {}
        for line in lines:
            loc = line.index(':')
            name = parse(line[:loc], style)
            for alias in line[loc+1:].strip().split(','):
                aliases[parse(alias,style)] = name
            aliases[name] = name
        return aliases

    aliases = {}
    current_host = ''
    for line in lines:
        loc = line.index(':')
        planet = parse(line[:loc], 'planet')
        host = parse(line[:loc], 'host')
        host_aliases = [
            parse(name, 'host')
            for name in line[loc+1:].strip().split(',')
        ]
        host_aliases += [host]
        planet_aliases = {
            parse(name, 'planet'): planet
            for name in line[loc+1:].strip().split(',')
        }
        planet_aliases[planet] = planet
        if host != current_host:
            # Save old one
            if current_host != '':
                system['host_aliases'] = np.unique(system['host_aliases'])
                aliases[current_host] = system
            # Start new one
            system = {
                'host': host,
                'planets': [planet],
                'host_aliases': host_aliases,
                'planet_aliases': planet_aliases,
            }
            current_host = host
        else:
            system['host_aliases'] += host_aliases
            system['planets'] += [planet]
            system['planet_aliases'].update(planet_aliases)
    system['host_aliases'] = np.unique(system['host_aliases'])
    aliases[current_host] = system
    return aliases


def find_target(targets=None):
    """
    Interactive prompt with tab-completion to search for targets.

    Parameters
    ----------
    targets: list of Target objects

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> target = cat.find_target()
    """
    if targets is None:
        targets = load_targets('nea_data.txt', is_confirmed=True)
        targets += load_targets('tess_data.txt', is_confirmed=False)
    planets = [target.planet for target in targets]
    aliases = []
    for target in targets:
        aliases += target.aliases
    planets += list(aliases)

    completer = ptk.completion.WordCompleter(
        planets,
        sentence=True,
        match_middle=True,
    )
    session = ptk.PromptSession(
        history=ptk.history.FileHistory(f'{ROOT}/data/target_search_history')
    )
    name = session.prompt(
        "(Press 'tab' for autocomplete)\nEnter Planet name: ",
        completer=completer,
        complete_while_typing=False,
    )
    if name in aliases:
        for target in targets:
            if name in target.aliases:
                return target
    if name in planets:
        return targets[planets.index(name)]

    return None


class Catalog():
    """
    Load the entire catalog.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> catalog = cat.Catalog()
    """
    def __init__(self, custom_targets=None):
        # Confirmed planets and TESS candidates
        nea_targets = load_targets('nea_data.txt', is_confirmed=True)
        tess_targets = load_targets('tess_data.txt', is_confirmed=False)
        self.targets = nea_targets + tess_targets
        if custom_targets is not None:
            custom = load_targets(custom_targets, is_confirmed=True)
            self.targets += custom

        # TBD: a switch between load_trexolists() and load_programs()?
        programs = load_trexolists(grouped=True)
        njwst = len(programs)
        aliases_file = f'{ROOT}data/target_aliases.txt'
        host_aliases = load_aliases('host', aliases_file)

        jwst_hosts = []
        for jwst_target in programs:
            host_names = [obs['target'] for obs in jwst_target]
            nea_host = np.unique([
                host_aliases[host] if host in host_aliases else host
                for host in host_names
            ])
            jwst_hosts += list(nea_host)
            for obs in jwst_target:
                obs['nea_host'] = nea_host
        jwst_hosts = np.unique(jwst_hosts)

        planet_aliases = load_aliases('planet', aliases_file)
        planets_aka = invert_aliases(planet_aliases)

        for target in self.targets:
            target.is_jwst_host = target.host in jwst_hosts
            if target.is_jwst_host:
                for j in range(njwst):
                    if target.host == programs[j][0]['nea_host']:
                        break
                target.programs = programs[j]
                planets = []
                for obs in programs[j]:
                    planets += obs['planets']
                planets = np.unique(planets)
                letter = get_letter(target.planet).strip()
                target.is_jwst_planet = letter in planets
            else:
                target.is_jwst_planet = False

            if target.planet in planets_aka:
                target.aliases = planets_aka[target.planet]

        self._transit_mask = [target.is_transiting for target in self.targets]
        self._jwst_mask = [target.is_jwst_host for target in self.targets]
        self._confirmed_mask = [target.is_confirmed for target in self.targets]

    def get_target(
            self, name=None,
            is_transit=True, is_jwst=None, is_confirmed=None,
        ):
        """
        Search by name for a planet in the catalog.

        Parameters
        ----------
        name: String
            If not None, name of the planet to search.
            If None, an interactive prompt will open to search for the planet
        is_transit: Bool
            If True/False restrict search to transiting/non-transiting planets
            If None, consider all targets.
        is_jwst: Bool
            If True/False restrict search to planet of/not JWST hosts
            If None, consider all targets.
        is_confirmed: Bool
            If True/False restrict search to confirmed/candidate planets
            If None, consider all targets.

        Returns
        -------
        target: a Target object
            Target with the system properties of the searched planet.
            If no target was found on the catalog, return None.
        """
        mask = np.ones(len(self.targets), bool)
        if is_transit is not None:
            mask &= np.array(self._transit_mask) == is_transit
        if is_jwst is not None:
            mask &= np.array(self._jwst_mask) == is_jwst
        if is_confirmed is not None:
            mask &= np.array(self._confirmed_mask) == is_confirmed

        targets = [target for target,flag in zip(self.targets,mask) if flag]

        if name is None:
            return find_target(targets)

        target = normalize_name(name)
        for target in targets:
            if name == target.planet or name in target.aliases:
                return target

    def show_target(
            self, name=None,
            is_transit=True, is_jwst=None, is_confirmed=True,
        ):
        target = self.get_target(name, is_transit, is_jwst, is_confirmed)
        if target is None:
            return
        print(target)

