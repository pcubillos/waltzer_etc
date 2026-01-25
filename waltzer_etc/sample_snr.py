# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'waltzer_sample',
]

import pickle

import numpy as np
import pandas as pd
import pyratbay.spectrum as ps
from .snr_waltzer import Detector
from . import sed
from .utils import inst_convolution


def waltzer_sample(
         csv_file,
         output_csv="waltzer_snr.csv",
         diameter=35.0,
         efficiency=0.6,
         t_dur=None,
         n_obs=10,
    ):
    """
    WALTzER Exposure time calculator

    Parameters
    ----------
    csv_file: String
        A .csv list of targets (e.g., downloaded from the NASA
        Exoplanet Archive).  This file must contain these headers:
        - 'pl_name'  Target's name
        - 'pl_trandur'  Transit duration (h)
        - 'st_teff'  Host's effective temperature (K)
        - 'st_rad'   Host's radius (R_sun)
        - 'st_mass'  Host's mass (M_sun)
        - 'ra'       Right ascention (degrees)
        - 'dec'      Declination (degrees)
        - 'sy_vmag'  Host's V magnitude
    output_csv: String
        Output filename where to save the resuls (as CSV)
        - planet               Target name
        - teff                 Host's effective temperature (K)
        - r_star               Host's radius (R_sun)
        - m_star               Host's mass (R_sun)
        - ra                   Right ascention (degrees)
        - dec                  Declination (degrees)
        - V_mag                Host's V magnitude
        - NUV_mean_flux        Mean flux in NUV band (erg s-1 cm-2 A-1)
        - NUV_max_flux         Maximum flux in NUV band (erg s-1 cm-2 A-1)
        - NUV_mean_snr         Mean SNR in NUV band
        - NUV_max_snr          Maximum SNR in NUV band
        - NUV_transit_uncert   NUV transit uncertainty
        - VIS_mean_flux        Mean flux in VIS band (erg s-1 cm-2 A-1)
        - VIS_max_flux         Maximum flux in VIS band (erg s-1 cm-2 A-1)
        - VIS_mean_snr         Mean SNR in VIS band
        - VIS_max_snr          Maximum SNR in VIS band
        - VIS_transit_uncert   VIS transit uncertainty
        - NIR_mean_flux        Mean flux in NIR band (erg s-1 cm-2 A-1)
        - NIR_max_flux         Maximum flux in NIR band (erg s-1 cm-2 A-1)
        - NIR_snr              SNR in NIR band
        - NIR_transit_uncert   NIR transit uncertainty
    TBD

    Notes
    -----
    Can also be run from the command line, e.g.:
    waltz target_list_20250327.csv waltzer_snr.csv

    Example (TBD while we build this machine)
    -------
    >>> import waltzer_etc as w
    >>> from waltzer_etc.snr_waltzer import Detector
    >>> from waltzer_etc.snr_waltzer import *
    >>> from waltzer_etc.utils import ROOT
    >>> import waltzer_etc.sed as sed
    >>> diameter = 35.0
    >>> csv_file = 'target_list_20250327.csv'
    >>> efficiency = 0.6
    >>> t_dur = 2.5
    >>> n_obs = 10
    """
    # The three amigos
    nuv_det = Detector('nuv', diameter)
    vis_det = Detector('vis', diameter)
    nir_det = Detector('nir', diameter)
    detectors = nuv_det, vis_det, nir_det
    bands = [det.band for det in detectors]

    # WALTzER resolution (FWHM) and wavelength grid (angstrom)
    inst_resolution = vis_det.resolution

    # Higher resolution for models (will be binned down to WALTzER later)
    resolution = 60_000.0
    wl = ps.constant_resolution_spectrum(0.23, 2.0, resolution=resolution)

    # Target list file path
    data = pd.read_csv(csv_file, delimiter=',', comment='#')
    ntargets = len(data)

    # Extract required columns into individual arrays
    planet_names = data['pl_name']
    transit_dur = data['pl_trandur']
    stellar_temps = data['st_teff']
    stellar_radii = data['st_rad']
    stellar_masses = data['st_mass']
    ra = data['ra']
    dec = data['dec']
    v_mags = data['sy_vmag']

    # Total in-transit exposure time (s) [for .csv statistics]
    if t_dur is None:
        # Use CSV data for each target
        exp_time = (transit_dur * 3600) * efficiency
    else:
        # Use fixed transit duration for all targets
        exp_time = (t_dur * 3600) * efficiency
        exp_time = np.tile(exp_time, ntargets)

    # Read models files
    cache_seds = {}
    output_data = []
    spectra = {}
    print('Target  Name           V_mag   Teff       NUV     VIS     NIR error (ppm)')
    for i in range(ntargets):
        target = planet_names[i]
        # Load SED model spectrum based on temperature
        teff = stellar_temps[i]
        logg = 4.5
        # TBD: sed_type hardcoded for now
        sed_type = 'llmodels'
        sed_file, sed_label, teff_match, logg_match = sed.find_closest_sed(teff, logg, sed_type)
        if sed_file in cache_seds:
            sed_flux = cache_seds[sed_file]
        else:
            # Load SED flux
            sed_wl, flux = sed.load_sed(teff_match, logg_match, sed_type)
            # Interpolate to regular grid and apply waltzer resolution
            flux = np.interp(wl, sed_wl, flux)
            sed_flux = inst_convolution(
                wl, flux, inst_resolution, sampling_res=resolution,
            )
            cache_seds[sed_file] = sed_flux

        # Normalize according to Vmag
        flux = sed.normalize_vega(wl, sed_flux, v_mags[i])

        # Flux stats
        nuv_flux_stats = nuv_det.flux_stats(wl, flux)
        vis_flux_stats = vis_det.flux_stats(wl, flux)
        nir_flux_stats = nir_det.flux_stats(wl, flux)

        # SNR stats
        integ_time = exp_time[i] * n_obs
        nuv_snr_stats = nuv_det.snr_stats(wl, flux, integ_time)
        vis_snr_stats = vis_det.snr_stats(wl, flux, integ_time)
        nir_snr_stats = nir_det.snr_stats(wl, flux, integ_time)[2:]
        print(
            f'{i+1:2d}/{ntargets}: {repr(target):15} '
            f'{v_mags[i]:5.2f}  {teff_match:5.0f}  '
            f'{nuv_snr_stats[-1]:8.1f}  {vis_snr_stats[-1]:6.1f}  '
            f'{nir_snr_stats[-1]:6.1f}'
        )

        # Note this should mirror GUI's run_waltzer()
        tso = {}
        for j,det in enumerate(detectors):
            band = det.band
            # Source flux and variance in e- per second:
            variances = det.calc_noise(wl, flux)
            total_variance = np.sum(variances, axis=0)
            band_flux = variances[0]
            tso[band] = {
                'wl': det.wl,
                'flux': band_flux,
                'variance': total_variance,
                'variances': variances,
                'det_type': det.mode,
                'half_widths': det.half_widths,
                'wl_min': det.wl_min,
                'wl_max': det.wl_max,
            }

        tso['meta'] = {
            'bands': bands,
            'efficiency': efficiency,
            'n_obs': n_obs,
            'transit_dur': transit_dur[i],
            'target': target,
        }

        spectra[target] = tso

        # Save results
        output_data.append([
            planet_names[i],
            stellar_temps[i],
            stellar_radii[i],
            stellar_masses[i],
            ra[i],
            dec[i],
            v_mags[i],
            transit_dur[i],
            *nuv_flux_stats,
            *nuv_snr_stats,
            *vis_flux_stats,
            *vis_snr_stats,
            *nir_flux_stats,
            *nir_snr_stats,
        ])

    # Save result as CSV file
    header = (
        "target, teff, r_star, m_star, ra, dec, V_mag, transit_dur, "
        "NUV_min_flux, NUV_median_flux, NUV_max_flux, "
        "NUV_min_snr, NUV_median_snr, NUV_max_snr, NUV_transit_uncert, "
        "VIS_min_flux, VIS_median_flux, VIS_max_flux, "
        "VIS_min_snr, VIS_median_snr, VIS_max_snr, VIS_transit_uncert, "
        "NIR_min_flux, NIR_median_flux, NIR_max_flux, "
        "NIR_snr, NIR_transit_uncert "
    ).split(',')
    header = [h.strip() for h in header]
    units = (
        "#name, K, R_sun, M_sun, deg, deg, , h,"
        "mJy, mJy, mJy, , , , ppm,"
        "mJy, mJy, mJy, , , , ppm,"
        "mJy, mJy, mJy, , ppm"
    ).split(',')

    with open(output_csv, 'w') as f:
        # Telemetry and stats info
        f.write(f"# primary diameter: {diameter} cm\n")
        f.write(f"# duty-cycle efficiency: {efficiency}\n")
        f.write(f"# instrumental resolution = lambda/FWHM = {inst_resolution}\n")

        f.write("#\n# SNR stats are per HWHM! resolving element\n")
        f.write(f"# Number of transits (for stats): {n_obs}\n")
        if t_dur is None:
            f.write("# total in-transit time (for stats): see 'transit_dur' column\n")
        else:
            f.write(f"# total in-transit time (for stats): {t_dur} h\n")

        f.write("#\n# Wavelength ranges per band (angstrom)\n")
        f.write(f"# NUV  {nuv_det.wl_min} {nuv_det.wl_max}\n")
        f.write(f"# VIS  {vis_det.wl_min} {vis_det.wl_max}\n")
        f.write(f"# NIR  {nir_det.wl_min} {nir_det.wl_max}\n#\n")
        f.write(','.join(header) + '\n')
        f.write(', '.join(units) + '\n')
        np.savetxt(f, output_data, delimiter=",", fmt="%s")

    p_file = output_csv.replace('.csv', '.pickle')
    with open(p_file, 'wb') as handle:
        pickle.dump(spectra, handle, protocol=4)
