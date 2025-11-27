# Copyright (c) 2025 Sreejith and Patricio
# LICENSE TBD

__all__ = [
    'Detector',
    'simulate_spectrum',
    'waltzer_snr',
]

import configparser
import pickle
import random

import numpy as np
import pandas as pd
import pyratbay.constants as pc
import pyratbay.spectrum as ps
import scipy.interpolate as si

from .utils import ROOT
from . import sed


class Detector():
    def __init__(self, detector_cfg, wl_edges, eff_collecting_area):
        """
        detector_cfg = 'detectors/waltzer_nuv.cfg'
        det = Detector(detector_cfg)

        Parameters
        ----------
        bin_edges: 1D float array
            Edges of the instrumental bins (angstrom).
        eff_collecting_area: float
            Effective collecting area of the instrument (cm²).
        """
        config = configparser.ConfigParser()
        config.read(detector_cfg)
        det = config['detector']
        self.mode = 'photometry' if 'nir' in detector_cfg else 'spectroscopy'
        self.resolution = det.getfloat('resolution')
        self.pix_scale = det.getfloat('pix_scale')
        self.fwhm = det.getfloat('fwhm')
        self.dark = det.getfloat('dark')
        self.read_noise = det.getfloat('read_noise')
        self.eff_area = eff_collecting_area

        self.wl_min = det.getfloat('wl_min')
        self.wl_max = det.getfloat('wl_max')

        i_min = np.searchsorted(wl_edges, self.wl_min)
        i_max = np.searchsorted(wl_edges, self.wl_max)
        self._wl_edges = wl_edges[i_min:i_max]
        # Center and half-widths of wavelength bins (angstrom):
        self.wl = 0.5 * (wl_edges[i_min+1:i_max] + wl_edges[i_min:i_max-1])
        self.half_widths = self.wl - wl_edges[i_min:i_max-1]

        if self.mode == 'photometry':
            self.wl = np.array([0.5 * (self.wl_max + self.wl_min)])
            self.half_widths = np.array([0.5 * (self.wl_max - self.wl_min)])
        self.nwave = len(self.wl)

        # TBD: In future this might depend on {RA,dec} of targets
        # Background flux (erg s-1 cm-2 A-1 arcsec-2)
        wl_bkg, sky = np.loadtxt(f'{ROOT}/data/background.txt', unpack=True)
        self.bkg_model = si.interp1d(
            wl_bkg, sky, kind='slinear', bounds_error=False,
        )

    def photon_spectrum(self, wl, flux):
        """
        Compute spectra in photons per second.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (angstrom).
        flux: 1D float array
            Input flux spectrum (erg s⁻¹ cm⁻² angstrom⁻¹).
        bkg_flux: 1D float array
            Background flux spectrum (erg s⁻¹ cm⁻² angstrom⁻¹ pixel⁻¹).

        Returns
        -------
        bin_flux: 1D float array
            Integrated photon flux per bin (photons s⁻¹).
        bin_bkg: 1D float array
            Integrated background photon flux per bin (photons s⁻¹ pixel⁻¹).

        Examples
        --------
        >>> resolution = 60_000.0
        >>> wl = ps.constant_resolution_spectrum(2_450, 20_000, resolution)
        >>> inst_resolution = 3_000.0
        >>> bin_edges = ps.constant_resolution_spectrum(
        >>>     2_500, 20_000, 2.0*inst_resolution,
        >>> )

        >>> # Load stellar SED
        >>> file = './models/t06000g4.4/model.flx'
        >>> sed_wl, flux, _ = np.loadtxt(file, unpack=True)
        >>> sed_flux = np.interp(wl, sed_wl, flux)

        >>> # Flux at instrumental resolution, in erg s-1 cm-2 A-1 at Earth
        >>> conv_flux = ps.inst_convolution(
        >>>     wl, sed_flux, inst_resolution, sampling_res=resolution,
        >>> )
        >>> star_R = 1.18 * pc.rsun
        >>> star_dist = 48.3 * pc.parsec
        >>> flux = conv_flux * (
        >>>     (pc.c / pc.A) / (wl**2)
        >>>     * 4.0 * np.pi
        >>>     * (star_R/star_dist)**2.0
        >>> )

        >>> effective_area = 170.296
        >>> wl_min = 6700
        >>> wl_max = 7100
        """
        # Convert flux (erg s-1 cm-2 A-1) to photons s-1 A-1
        photons = flux * wl*pc.A / (pc.c*pc.h) * self.eff_area

        # Background flux (erg s-1 cm-2 A-1 pixel-1)
        bkg_flux = self.bkg_model(wl) * self.pix_scale**2.0
        # Convert flux (erg s-1 cm-2 A-1 pixel) to photons s-1 A-1 pixel-1
        bkg_photons = bkg_flux * wl*pc.A / (pc.c*pc.h) * self.eff_area

        # Integrate at each instrumental bin to get photons s-1
        if self.mode == 'photometry':
            mask = (wl>=self.wl_min) & (wl<=self.wl_max)
            bin_flux = np.trapezoid(photons[mask], wl[mask])
            bin_bkg = np.trapezoid(bkg_photons[mask], wl[mask])
            return (
                np.array([bin_flux]),
                np.array([bin_bkg]),
            )

        # Spectroscopy: integrate (photons s-1 A-1 pix-1) to photons s-1 pix-1
        wl_edges = self._wl_edges
        bin_flux = np.zeros(self.nwave)
        bin_bkg = np.zeros(self.nwave)
        for i in range(self.nwave):
            bin_mask = (
                (wl>=wl_edges[i]) &
                (wl<=wl_edges[i+1])
            )
            bin_flux[i] = np.trapezoid(photons[bin_mask], wl[bin_mask])
            bin_bkg[i] = np.trapezoid(bkg_photons[bin_mask], wl[bin_mask])

        return bin_flux, bin_bkg

    def flux_stats(self, wl, flux):
        """
        Compute basic flux statistics within a wavelength interval.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (angstrom).
        flux: 1D float array
            Input flux spectrum (erg s⁻¹ cm⁻² angstrom⁻¹).

        Returns
        -------
        mean_flux: float
            Mean flux within the wavelength range (erg s⁻¹ cm⁻² angstrom⁻¹).
        max_flux: float
            Maximum flux within the wavelength range (erg s⁻¹ cm⁻² angstrom⁻¹).
        """
        band = (wl > self.wl_min) & (wl < self.wl_max)
        mean_flux = np.mean(flux[band])
        max_flux = np.max(flux[band])
        return mean_flux, max_flux

    def snr_stats(self, flux, exp_time, n_obs=1, bkg_flux=None):
        """
        Compute basic SNR statistics within a wavelength interval.

        Parameters
        ----------
        flux: 1D float array
            Photon flux per spectral bin (photons s⁻¹).
        exp_time: Float
            In-transit exposure time (s).
        n_obs: Integer
            Number of transits.
        """
        # integrate over time
        total_flux = flux * exp_time * n_obs

        # Poisson noise estimation
        snr = total_flux / np.sqrt(total_flux)
        snr_mean = np.mean(snr)
        snr_max = np.max(snr)

        # Assume t_in = t_out
        # Assume flux_in approx flux_out
        trans_uncer = np.sqrt(2.0) / snr_mean / pc.ppm

        return snr_mean, snr_max, np.round(trans_uncer, 3)


def simulate_spectrum(
        tso, depth_model, n_obs=1, resolution=300.0,
        transit_dur=None, efficiency=0.6, noiseless=False,
    ):
    """
    Combine a WALTzER output SNR data with a transmission spectrum
    to simulate observations.

    Parameters
    ----------
    tso: Dictionary
        A planet's model, output from running snr_waltzer.py.
    depth_model: 2D float array
        Transit depth model, array of shape [2,nwave] with:
        - wavelength (angstrom)
        - transit depth
    n_obs: Integer
        Number of transits to co-add.
    resolution: Float
        Output resolution.
    transit_dur: Float
        If not None, overwrite transit duration (h) from tso dictionary.
    efficiency: Float
        WALTZER duty cycle efficiency.

    Returns
    -------
    walz_wl: List of 1D float arrays
        WALTZER wavelength array for NUV, VIS, and NIR bands (angstrom)
    walz_spec: List of 1D float arrays
        WALTZER transit depths for each band
    walz_err: List of 1D float arrays
        WALTZER transit-depth uncertainties for each band
    walz_widths: List of 1D float arrays
        Wavelength half-widths of WALTZER data points (angstrom).

    Examples
    --------
    >>> import pickle
    >>> import numpy as np
    >>> import pyratbay.constants as pc
    >>> import snr_waltzer as w

    >>> # Load WALTzER SNR output pickle file
    >>> tso_file = 'waltzer_snr_test.pickle'
    >>> with open(tso_file, 'rb') as handle:
    >>>     spectra = pickle.load(handle)
    >>> tso = spectra['HD 209458 b']

    >>> # Load a transit-depth spectrum
    >>> tdepth_file = 'transit_saturn_1600K_clear.dat'
    >>> depth_model = np.loadtxt(tdepth_file, unpack=True)

    >>> sim = w.simulate_spectrum(
    >>>     tso, depth_model,
    >>>     n_obs=10,
    >>>     resolution=300.0,
    >>>     noiseless=False,
    >>> )

    >>> waltzer_wl, waltzer_spec, waltzer_err, waltzer_widths = sim
    >>> fig = plt.figure(0)
    >>> plt.clf()
    >>> fig.set_size_inches(8,4)
    >>> plt.plot(depth_model[0], depth_model[1]/pc.percent, color='xkcd:blue')
    >>> bands = ['NUV', 'VIS', 'NIR']
    >>> for j,band in enumerate(bands):
    >>>     plt.errorbar(
    >>>         waltzer_wl[j], waltzer_spec[j]/pc.percent,
    >>>         waltzer_err[j]/pc.percent, xerr=waltzer_widths[j],
    >>>         fmt='o', ecolor='cornflowerblue', color='royalblue',
    >>>         mfc='w', ms=4, zorder=0,
    >>>     )
    >>> plt.xscale('log')
    >>> plt.xlim(2400, 17000)
    >>> plt.ylim(0.99, 1.12)
    """
    # Interpolate to WALTzER grid:
    model = si.interp1d(
        depth_model[0], depth_model[1],
        kind='slinear', bounds_error=False, fill_value='extrapolate',
    )

    if transit_dur is None:
        transit_dur = tso['transit_dur']

    dt_in = transit_dur * 3600.0
    dt_out = 2.0 * np.amax([0.5*dt_in, 3600.0])

    walz_wl = []
    walz_spec = []
    walz_err = []
    walz_widths = []
    for j in range(3):
        flux = tso['e_flux'][j]
        half_width = tso['wl_half_width'][j]
        wl = tso['wl'][j]

        # Transit
        depth = model(wl)
        flux_out = flux * n_obs * efficiency
        flux_in = (1.0-depth) * flux * n_obs * efficiency
        # Poisson noise estimation
        var_out = flux_out*dt_out
        var_in = flux_in*dt_in

        # Photometry
        if len(wl) == 1:
            bin_in = flux_in
            bin_out = flux_out
            bin_vin = var_in
            bin_vout = var_out
            bin_wl = wl
            bin_widths = half_width
            band = ps.Tophat(wl[0], half_width[0], wl=depth_model[0])
            bin_depth = band.integrate(depth_model[1])
        # Spectroscopy binning
        else:
            wl_min = np.amin(wl)
            wl_max = np.amax(wl)
            if resolution is not None:
                bin_edges = ps.constant_resolution_spectrum(
                    wl_min, wl_max, resolution,
                )
                bin_edges = np.append(bin_edges, wl_max)
            else:
                bin_edges = 0.5 * (wl[1:] + wl[:-1])

            bin_wl = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_widths = bin_wl - bin_edges[:-1]
            bin_depth = ps.bin_spectrum(bin_wl, wl, depth)
            nbins = len(bin_wl)
            bin_out = np.zeros(nbins)
            bin_in = np.zeros(nbins)
            bin_vout = np.zeros(nbins)
            bin_vin = np.zeros(nbins)
            for i in range(nbins):
                bin_mask = (wl>=bin_edges[i]) & (wl<bin_edges[i+1])
                bin_out[i] = np.sum(flux_out[bin_mask])
                bin_in[i] = np.sum(flux_in[bin_mask])
                bin_vout[i] = np.sum(var_out[bin_mask])
                bin_vin[i] = np.sum(var_in[bin_mask])

        bin_err = np.sqrt(
            (1.0/dt_in/bin_out)**2.0 * bin_vin +
            (bin_in/dt_out/bin_out**2.0)**2.0 * bin_vout
        )
        # The numpy random system must have its seed reinitialized in
        # each sub-processes to avoid identical 'random' steps.
        # random.randomint is process- and thread-safe.
        np.random.seed(random.randint(0, 100000))
        rand_noise = np.random.normal(0.0, bin_err)
        if noiseless:
            rand_noise *= 0.0

        bin_spec = bin_depth + rand_noise

        walz_wl.append(bin_wl)
        walz_spec.append(bin_spec)
        walz_err.append(bin_err)
        walz_widths.append(bin_widths)

    return walz_wl, walz_spec, walz_err, walz_widths


def waltzer_snr(
         csv_file=None,
         output_csv="waltzer_snr.csv",
         diameter=30.0,
         efficiency=0.6,
         t_dur=2.5,
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
        - 'sy_dist'  Distant to target (parsec)
        - 'sy_vmag'  Host's V magnitude
    output_csv: String
        Output filename where to save the resuls (as CSV)
        - planet               Target name
        - teff                 Host's effective temperature (K)
        - r_star               Host's radius (R_sun)
        - m_star               Host's mass (R_sun)
        - ra                   Right ascention (degrees)
        - dec                  Declination (degrees)
        - distance             Distant to target (parsec)
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
    python snr_waltzer.py target_list_20250327.csv waltzer_snr.csv

    Example (TBD while we build this machine)
    -------
    >>> import snr_waltzer as w
    >>> from waltzer_etc.snr_waltzer import *
    >>> from waltzer_etc.utils import ROOT
    >>> from waltzer_etc.normalization import normalize_vega
    >>> diameter = 30.0
    >>> csv_file = 'target_list_20250327.csv'
    >>> efficiency = 0.6
    >>> t_dur = 2.5
    >>> n_obs = 10
    """
    # Effective area calculations
    primary_area = np.pi * (0.5*diameter)**2.0
    Rprim    = 0.90  # Telescope primary reflectance in %
    Rsec     = 0.90  # Telescope secondary reflectance in %
    Sec_obstr= 0.85  # Telescope secondary obstruction in % (1-Obstruction)
    R_d1     = 0.80  # Dichroic 1 Reflectance/Transmission in %
    R_d2     = 0.80  # Dichroic 2 Reflectance/Transmission in %

    R_uvfold = 1.00  # UV fold reflectance in %
    R_uvgr   = 0.90  # UV grating reflectance in %
    Uv_geff  = 0.65  # UV grating effeciency in %
    Uv_detQE = 0.55  # UV detector QE in %

    R_opfold = 0.90  # Optical fold reflectance in %
    R_opgr   = 0.90  # Optical grating reflectance in %
    Op_geff  = 0.75  # Optical grating effeciency in %
    Op_detQE = 0.90  # Optical detector QE in %

    BB_lens  = 0.90  # IR Broad band lens transmission in % IN REALITY, A FOLD
    BB_detQE = 0.85  # IR Broad band detector QE in %

    # Effective collecing areas in cm^2
    NUV_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_uvfold * R_uvgr * Uv_geff * Uv_detQE
    )
    VIS_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_d2 * R_opfold * R_opgr * Op_geff * Op_detQE
    )
    NIR_EFF_AREA = (
        primary_area * Rprim * Rsec * Sec_obstr *
        R_d1 * R_d2 * BB_lens * BB_detQE
    )

    # WALTzER resolution (FWHM) and wavelength grid (angstrom)
    inst_resolution = 3_000.0
    bin_edges = ps.constant_resolution_spectrum(
        2_500, 20_000, resolution=2.0*inst_resolution,
    )

    detector_cfg = f'{ROOT}/data/detectors/waltzer_nuv.cfg'
    nuv_det = Detector(detector_cfg, bin_edges, NUV_EFF_AREA)
    detector_cfg = f'{ROOT}/data/detectors/waltzer_vis.cfg'
    vis_det = Detector(detector_cfg, bin_edges, VIS_EFF_AREA)
    detector_cfg = f'{ROOT}/data/detectors/waltzer_nir.cfg'
    nir_det = Detector(detector_cfg, bin_edges, NIR_EFF_AREA)

    # Higher resolution for models (will be bin down to WALTzER)
    resolution = 60_000.0
    wl = ps.constant_resolution_spectrum(2_450, 20_000, resolution=resolution)

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
    distances = data['sy_dist']
    v_mags = data['sy_vmag']

    # Total in-transit exposure time (s) [for .csv statistics]
    if t_dur is None:
        # Use CSV data for each target
        exp_time = (transit_dur * 3600) * efficiency
    else:
        # Use fixed transit duration for all targets
        exp_time = (t_dur * 3600) * efficiency
        exp_time = np.tile(exp_time, ntargets)

        print('Target Name  V_mag  Teff  SNR_NUV SNR_VIS SNR_NIR (ppm)')
    # Read models files
    cache_seds = {}
    output_data = []
    spectra = {}
    print('Target  Name          V_mag   Teff  SNR: NUV     VIS     NIR (ppm)')
    for i in range(ntargets):
        target = planet_names[i]
        # Load SED model spectrum based on temperature
        teff = stellar_temps[i]
        sed_file, teff_match = sed.find_closest_teff(teff)
        if sed_file in cache_seds:
            sed_flux = cache_seds[sed_file]
        else:
            # Load SED flux
            sed_wl, flux = sed.load_sed(file=sed_file)
            # Interpolate to regular grid and apply waltzer resolution
            flux = np.interp(wl, sed_wl, flux)
            sed_flux = ps.inst_convolution(
                wl, flux, inst_resolution, sampling_res=resolution,
            )
            cache_seds[sed_file] = sed_flux

        # Normalize according to Vmag
        flux = sed.normalize_vega(wl, sed_flux, v_mags[i])

        nuv_flux_stats = nuv_det.flux_stats(wl, flux)
        vis_flux_stats = vis_det.flux_stats(wl, flux)
        nir_flux_stats = nir_det.flux_stats(wl, flux)

        # Fluxes in photons per second
        nuv_flux, nuv_background = nuv_det.photon_spectrum(wl, flux)
        vis_flux, vis_background = vis_det.photon_spectrum(wl, flux)
        nir_flux, nir_background = nir_det.photon_spectrum(wl, flux)

        # snr_stats(flux, exp_time, n_obs, bkg_flux=None)
        nuv_snr_stats = nuv_det.snr_stats(nuv_flux, exp_time[i], n_obs)
        vis_snr_stats = vis_det.snr_stats(vis_flux, exp_time[i], n_obs)
        nir_snr_stats = nir_det.snr_stats(nir_flux, exp_time[i], n_obs)[1:]
        print(
            f'{i+1:2d}/{ntargets}: {repr(target):15} '
            f'{v_mags[i]:4.2f}  {teff_match:5.0f}  '
            f'{nuv_snr_stats[-1]:8.1f}  {vis_snr_stats[-1]:6.1f}  '
            f'{nir_snr_stats[-1]:6.1f}'
        )

        wl_half_widths = (
            nuv_det.half_widths,
            vis_det.half_widths,
            nir_det.half_widths,
        )
        spectra[target] = {
            'wl': (nuv_det.wl, vis_det.wl, nir_det.wl),
            'e_flux': (nuv_flux, vis_flux, nir_flux),
            'wl_half_width': wl_half_widths,
            'transit_dur': transit_dur[i],
        }

        # Save results
        output_data.append([
            planet_names[i],
            stellar_temps[i],
            stellar_radii[i],
            stellar_masses[i],
            ra[i],
            dec[i],
            distances[i],
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
        "target, teff, r_star, m_star, ra, dec, distance, V_mag, transit_dur,"
        "NUV_mean_flux, NUV_max_flux,"
        "NUV_mean_snr, NUV_max_snr, NUV_transit_uncert, "
        "VIS_mean_flux, VIS_max_flux, "
        "VIS_mean_snr, VIS_max_snr, VIS_transit_uncert, "
        "NIR_mean_flux, NIR_max_flux, NIR_snr, NIR_transit_uncert "
    ).split(',')
    header = [h.strip() for h in header]
    units = (
        "#name, K, R_sun, M_sun, deg, deg, parsec, , h,"
        "erg s-1 cm-2 A-1, erg s-1 cm-2 A-1, , , ppm,"
        "erg s-1 cm-2 A-1, erg s-1 cm-2 A-1, , , ppm,"
        "erg s-1 cm-2 A-1, erg s-1 cm-2 A-1, , ppm"
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
