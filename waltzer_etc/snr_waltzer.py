# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'Detector',
    'calc_variances',
    'bin_tso_data',
    'simulate_fluxes',
    'simulate_spectrum',
]

import configparser
import os
import random

import numpy as np
import pyratbay.constants as pc
import pyratbay.spectrum as ps
import scipy.interpolate as si

from .utils import ROOT, PassBand, inst_convolution


DEFAULT_DETS = {
    'nuv': f'{ROOT}/data/detectors/waltzer_nuv.cfg',
    'vis': f'{ROOT}/data/detectors/waltzer_vis.cfg',
    'nir': f'{ROOT}/data/detectors/waltzer_nir.cfg',
}


def calc_collecting_area(diameter, band):
    """
    Calculate the effective collecting area of the telescope+detector
    (everything except the quantum efficiencies)

    No longer used, but might come back to life in the future.
    """
    primary_area = np.pi * (0.5*diameter)**2.0
    Rprim    = 0.90  # Telescope primary reflectance in %
    Rsec     = 0.90  # Telescope secondary reflectance in %
    Sec_obstr= 0.85  # Telescope secondary obstruction in % (1-Obstruction)
    R_d1     = 0.80  # Dichroic 1 Reflectance/Transmission in %
    R_d2     = 0.80  # Dichroic 2 Reflectance/Transmission in %

    R_uvfold = 0.87  # UV fold reflectance in %
    R_uvgr   = 0.87  # UV grating reflectance in %
    Uv_geff  = 0.65  # UV grating effeciency in %

    R_opfold = 0.90  # Optical fold reflectance in %
    R_opgr   = 0.90  # Optical grating reflectance in %
    Op_geff  = 0.75  # Optical grating effeciency in %

    # Effective collecing areas in cm^2
    if band == 'nuv':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_uvfold * R_uvgr * Uv_geff
        )
        return eff_area

    if band == 'vis':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_d2 * R_opfold**2 * R_opgr * Op_geff
        )
        return eff_area

    if band == 'nir':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_d2 * R_opfold
        )
        return eff_area

    raise ValueError(f'Invalid band {repr(band)}')


def throughput(file, primary_area=1.0):
    """
    Load a detector throughput curve into a scipy interpolator function

    Parameters
    ----------
    file: String
        Path to a tabulated throughput response function.
        Wavelenth in first column (um), throughput in last column.
    primary_area: Float
        Area of primary mirror in cm^2.
    Returns
    -------
    min_wl: Float
    max_wl: Float
    throughput: scipy interpolator
    """
    if not os.path.exists(file):
        raise ValueError(f'Quantum-efficiency file not found: {repr(file)}')
    det_response = np.loadtxt(file, unpack=True)
    wl = det_response[0]
    wl_min = np.amin(wl)
    wl_max = np.amax(wl)
    response = det_response[-1] * primary_area

    fill_value = response[0], response[-1]
    throughput = si.interp1d(
        wl, response,
        kind='slinear',
        bounds_error=False, fill_value=fill_value,
    )
    return wl_min, wl_max, throughput


class Detector():
    def __init__(self, detector_cfg, diameter=35.0):
        """
        detector_cfg = 'detectors/waltzer_nuv.cfg'
        det = Detector(detector_cfg)

        Parameters
        ----------
        detector_cfg: string
            Path to a detector configuration file.
            Or 'nuv', 'vis', 'nir' to use one of the default configurations.
        diameter: float
            Telescope primary-mirror diameter (cm).

        Examples
        --------
        >>> import waltzer_etc as waltz
        >>> det = waltz.Detector('nuv')
        """
        if detector_cfg in DEFAULT_DETS:
            detector_cfg = DEFAULT_DETS[detector_cfg]

        config = configparser.ConfigParser()
        config.read(detector_cfg)
        det = config['detector']
        self.band = det.get('band')
        self.mode = det.get('mode')

        self.resolution = det.getfloat('resolution')
        self.pix_scale = det.getfloat('pix_scale')
        self.dark = det.getfloat('dark')
        self.read_noise = det.getfloat('read_noise')
        self.exp_time = det.getfloat('exp_time')
        self.aperture = det.getint('aperture')

        # Number of pixels in source and in sky-background
        aperture_radius = self.aperture // 2
        if self.mode == 'photometry':
            self.npix = int(np.pi*aperture_radius**2)
            sky_in = aperture_radius
            sky_out = 2*aperture_radius
            self.nsky = int(np.pi * (sky_out**2 - sky_in**2))
        elif self.mode == 'spectroscopy':
            self.npix = 2*aperture_radius
            sky_in = aperture_radius
            sky_out = 3*aperture_radius
            self.nsky = 2 * (sky_out-sky_in)

        self.primary_area = np.pi * (0.5*diameter)**2.0
        throughput_file = f'{ROOT}/data/detectors/{det.get("throughput_file")}'
        self.wl_min, self.wl_max, self.throughput = throughput(
            throughput_file, self.primary_area,
        )
        margin_left = 2*self.wl_min/self.resolution
        margin_right = 2*self.wl_max/self.resolution

        # High-resolution sampling
        over_sampling = 8
        self.hires_resolution = 2.0*self.resolution * over_sampling
        self.hires_wl_min = 0.23
        self.hires_wl_max = 2.0
        hires_wl = ps.constant_resolution_spectrum(
            self.hires_wl_min, self.hires_wl_max, self.hires_resolution,
        )

        wl_min = self.wl_min - margin_left
        wl_max = self.wl_max + margin_right
        self.hires_wl = hires_wl[
            (hires_wl>=wl_min) &
            (hires_wl<=wl_max)
        ]

        i_wl_ini = np.searchsorted(self.hires_wl, self.wl_min)
        i_wl_end = np.searchsorted(self.hires_wl, self.wl_max)
        wl_edges = self.hires_wl[i_wl_ini:i_wl_end:over_sampling]
        # Center and half-widths of wavelength bins (micron):
        self.wl = 0.5 * (wl_edges[1:] + wl_edges[:-1])
        self.half_widths = 0.5 * (wl_edges[1:] - wl_edges[:-1])

        # Sampling conversion
        self._wl_edges = wl_edges
        self.i_start = i_wl_ini
        self.nwave = len(wl_edges) - 1
        self.over_sampling = over_sampling

        if self.mode == 'photometry':
            self.wl = np.array([0.5 * (self.wl_max + self.wl_min)])
            self.half_widths = np.array([0.5 * (self.wl_max - self.wl_min)])
        self.nwave = len(self.wl)

        # TBD: In future this might depend on {RA,dec} of targets
        # Background flux (erg s-1 cm-2 A-1 arcsec-2)
        wl_bkg, sky = np.loadtxt(f'{ROOT}/data/background.txt', unpack=True)
        # Convert wavelength from angstrom to micron:
        wl_bkg /= pc.um/pc.A
        self.bkg_model = si.interp1d(
            wl_bkg, sky, kind='slinear', bounds_error=False,
        )


    def photon_spectrum(self, wl, flux, readout='full_frame'):
        """
        Compute spectra in photons per second.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (micron).
        flux: 1D float array
            Input flux spectrum (mJy).

        Returns
        -------
        bin_flux: 1D float array
            Integrated photon flux per bin (photons s⁻¹).
        bin_bkg: 1D float array
            Integrated background photon flux per bin (photons s⁻¹ pixel⁻¹).

        Examples
        --------
        >>> # TBD
        >>> resolution = 60_000.0
        >>> wl = ps.constant_resolution_spectrum(0.23, 2.0, resolution)
        >>> inst_resolution = 3_000.0
        >>> bin_edges = ps.constant_resolution_spectrum(
        >>>     2_400, 20_000, 2.0*inst_resolution,
        >>> )

        >>> # Load stellar SED
        >>> file = './models/t06000g4.4/model.flx'
        >>> sed_wl, flux, _ = np.loadtxt(file, unpack=True)
        >>> sed_flux = np.interp(wl, sed_wl, flux)

        >>> # Flux at instrumental resolution, in erg s-1 cm-2 A-1 at Earth
        >>> conv_flux = inst_convolution(
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
        wl_min = np.amin(self.hires_wl)
        wl_max = np.amax(self.hires_wl)
        ini = np.searchsorted(wl, wl_min)
        end = np.searchsorted(wl, wl_max) + 1

        flux = flux[ini:end]
        wl = wl[ini:end]

        throughput = self.throughput(wl)
        # Convert flux (mJy) to photons s-1 Hz-1
        photon_energy = pc.h * pc.c / (wl*pc.um)
        photons_nu = 1e-26 * flux / photon_energy * throughput
        # Now convert to photons s-1 um-1
        photons = photons_nu * pc.c / (wl*pc.um)**2.0 * pc.um

        # Background flux (erg s-1 cm-2 A-1 pixel-1)
        bkg_flux = self.bkg_model(wl) * self.pix_scale**2.0
        # Convert flux (erg s-1 cm-2 A-1 pixel-1) to photons s-1 um-1 pixel-1
        bkg_photons = bkg_flux / photon_energy * throughput * pc.um/pc.A

        self.e_flux = photons
        self.e_background = bkg_photons
        return photons, bkg_photons


    def flux_stats(self, wl, flux):
        """
        Compute basic flux statistics within a wavelength interval.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (microns).
        flux: 1D float array
            Input flux spectrum (mJy).

        Returns
        -------
        min_flux: float
            Minimum flux within the wavelength range (mJy).
        median_flux: float
            Mean flux within the wavelength range (mJy).
        max_flux: float
            Maximum flux within the wavelength range (mJy).
        """
        wl_min = np.amin(self.hires_wl)
        wl_max = np.amax(self.hires_wl)
        band = (wl >= wl_min) & (wl <= wl_max)

        sed_flux = inst_convolution(
            wl[band], flux[band], self.resolution, self.hires_resolution,
        )

        min_flux = np.min(sed_flux)
        median_flux = np.median(sed_flux)
        max_flux = np.max(sed_flux)
        return min_flux, median_flux, max_flux


    def snr_stats(self, var_data, integ_time):
        """
        Compute basic SNR statistics.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (micron).
        flux: 1D float array
            Input flux spectrum (mJy).
        integ_time: Float
            In-transit exposure time (s).

        Returns
        -------
        snr_min: Float
            Min signal-to-noise of the flux measurement.
        snr_median: Float
            Mean signal-to-noise of the flux measurement.
        snr_max: Float
            Max signal-to-noise of the flux measurement.
        transit_uncert: Float
            Mean transit-depth uncertainty (ppm) assuming tdur=integ_time.
        """
        source = var_data[2]
        variance = np.sum(np.array(var_data[2:]), axis=0)

        # Signal-to-noise estimation
        snr = source / np.sqrt(variance) * np.sqrt(integ_time)
        snr_min = np.min(snr)
        snr_median = np.median(snr)
        snr_max = np.max(snr)

        # Assume t_in = t_out
        # Assume flux_in approx flux_out
        transit_uncert = np.sqrt(2.0) / snr_median / pc.ppm

        return snr_min, snr_median, snr_max, transit_uncert


def calc_variances(tso, readout='full_frame', exp_time=300.0):
    """
    Compute the electrons per second signal of the source,
    background, dark, and read noise.

    Convolve to WALTzER resolving power (FWHM=3000),
    Integrate signals over WALTzER pixels.

    Parameters
    ----------
    wl: 1D float array
        Input wavelength array (micron).
    flux: 1D float array
        Input flux spectrum (mJy).
    integ_time: Float
        Total integration time (s).

    Returns
    -------
    wl: 1D float array
        Wavelength array at WALTzER pixel positions.
    half_width: 1D float array
        Wavelength half width of WALTzER pixels.
    var_source: 1D float array
        Source Poisson noise variance.
    var_background: 1D float array
        Background Poisson noise variance.
    var_dark: 1D float array
        Dark-current variance.
    var_read: 1D float array
        Read-noise variance.
    """
    npix = tso['npix']
    nsky = tso['nsky']
    hires_wl = tso['hires_wl']
    background = tso['background']
    # For the time being, ignore the fact that nreads should be integer
    nreads = 1.0 / exp_time

    # Convolve SED flux to WALTzER resolving power
    flux = inst_convolution(
        hires_wl, tso['flux'], tso['resolution'],  tso['hires_resolution'],
    )

    # Integrate at each instrumental pixel to get photons s-1
    if tso['det_type'] == 'photometry':
        bin_flux = np.array([np.trapezoid(flux, hires_wl)])
        bin_bkg = np.array([np.trapezoid(background, hires_wl)])
        nwave = 1
        rebin = 1
        wl = [0.5 * (tso['wl_max']+tso['wl_min'])]
        half_widths = [0.5 * (tso['wl_max']-tso['wl_min'])]

    elif tso['det_type'] == 'spectroscopy':
        rebin = 1
        if readout == 'faint':
            rebin = 2
        elif readout == 'ultra_faint':
            rebin = 4
        nwave = tso['nwave'] // rebin
        binsize = tso['over_sampling'] * rebin

        # Fluxes in photons per second
        bin_flux = np.zeros(nwave)
        bin_bkg = np.zeros(nwave)
        i_start = tso['i_start']
        for i in range(nwave):
            i1 = i_start + i*binsize
            i2 = i1 + binsize + 1
            bin_flux[i] = np.trapezoid(flux[i1:i2], hires_wl[i1:i2])
            bin_bkg[i] = np.trapezoid(background[i1:i2], hires_wl[i1:i2])

        i_end = i_start + (nwave+1)*binsize
        wl_edges = hires_wl[i_start:i_end:binsize]
        # Center and half-widths of wavelength bins:
        wl = 0.5 * (wl_edges[1:] + wl_edges[:-1])
        half_widths = 0.5 * (wl_edges[1:] - wl_edges[:-1])

    # Integrate over time
    var_source = np.abs(bin_flux)

    # Background number of photons
    var_background = npix*(1+npix/nsky) * bin_bkg

    # Dark number of photons
    var_dark = rebin * npix*(1+npix/nsky) * tso['dark']
    var_dark = np.tile(var_dark, nwave)

    # Read-out noise
    if readout == 'full_frame' or tso['det_type'] == 'photometry':
        var_read = npix*(1+npix/nsky) * tso['read_noise'] * nreads

    elif readout in ['bright', 'faint', 'ultra_faint']:
        # Only two reads (instead of nsky) for the background:
        var_read = npix*(1+2*npix/nsky**2) * tso['read_noise'] * nreads

    var_read = np.tile(var_read, nwave)

    return (
        wl,
        half_widths,
        var_source,
        var_background,
        var_dark,
        var_read,
    )


def bin_tso_data(
        wl, flux, var, throughput, half_width,
        depth_model, obs_type, dt_in, dt_out,
        det_type, binsize=None, resolution=None,
    ):
    """
    Compute binned in- and out-of-transit spectra and their variances
    for WALTzER observations.

    Parameters
    ----------
    depth_model: scipy.interpolate.interp1d() object
        Transit depth model as function of wavelength (micron).
    obs_type: string
        The observing geometry 'transit' or 'eclipse', or 'stare'.
    dt_in: Float
        In-transit integration time (seconds).
    dt_out: Float
        Out-of-transit (or stare) integration time (seconds).
    det_type: string
        Detector type: 'spectroscopy' or 'photometry'.
    binsize: Integer
        Alternative to resolution binning, binsize indicates to bin
        by number of pixels.
    resolution: Float
        Output resolution.
    """
    # Evaluate transit-depth in the band:
    if det_type == 'photometry':
        band_data = wl, throughput
        band = PassBand(band_data, wl=depth_model.x)
        if band.wl is None:
            depth = np.nan
        else:
            depth = band.integrate(depth_model.y)
        wl = np.array([band.wl0])
    else:
        # TBD: convolve, integrate instead
        depth = depth_model(wl)

    if obs_type == 'transit':
        # Fluxes [e- collected] in and out of transit
        flux_out = flux * dt_out
        flux_in = flux * (1.0-depth) * dt_in
        # Variance estimations (e- collected) in and out of transit
        var_out = var * dt_out
        var_in = var * (1.0-depth) * dt_in

    elif obs_type == 'eclipse':
        # Fluxes [e- collected] in and out of eclipse
        flux_out = flux * (1.0+depth) * dt_out
        flux_in = flux * dt_in
        # Variance estimations (e- collected) in and out of eclipse
        var_out = var * (1.0+depth) * dt_out
        var_in = var * dt_in

    elif obs_type == 'stare':
        # Fluxes [e- collected] in and out of eclipse
        flux_out = flux * dt_out
        flux_in = flux * 0.0
        # Variance estimations (e- collected) in and out of eclipse
        var_out = var * dt_out
        var_in = var * 0.0

    if binsize is None and resolution is None:
        binsize = 1
    elif resolution == 0.0:
       binsize = 1

    # Photometry
    no_binning = (
        det_type == 'photometry'
        or binsize == 1
    )

    if no_binning:
        return wl, flux_in, flux_out, var_in, var_out, half_width, depth

    # Bin by binsize
    if resolution is None:
        nwave = len(wl)
        bin_idx = np.arange(0, nwave, binsize)
        counts = np.diff(np.append(bin_idx, nwave))

        bin_widths = np.add.reduceat(half_width, bin_idx)
        bin_wl = np.add.reduceat(wl, bin_idx) / counts
        bin_depth = np.add.reduceat(depth, bin_idx) / counts

        bin_in = np.add.reduceat(flux_in, bin_idx)
        bin_out = np.add.reduceat(flux_out, bin_idx)
        bin_var_in = np.add.reduceat(var_in, bin_idx)
        bin_var_out = np.add.reduceat(var_out, bin_idx)

    # Bin by resolution
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
        bin_var_out = np.zeros(nbins)
        bin_var_in = np.zeros(nbins)
        for i in range(nbins):
            bin_mask = (wl>=bin_edges[i]) & (wl<bin_edges[i+1])
            bin_out[i] = np.sum(flux_out[bin_mask])
            bin_in[i] = np.sum(flux_in[bin_mask])
            bin_var_out[i] = np.sum(var_out[bin_mask])
            bin_var_in[i] = np.sum(var_in[bin_mask])

    return (
        bin_wl,
        bin_in,
        bin_out,
        bin_var_in,
        bin_var_out,
        bin_widths,
        bin_depth,
    )


def simulate_fluxes(
        tso, depth_model=None, obs_type='transit', n_obs=1,
        transit_dur=None, obs_dur=None,
        binsize=None, resolution=None, noiseless=False,
        efficiency=None,
    ):
    """
    Simulate WALTzER TSO fluxes and variances during an observation.

    Parameters
    ----------
    tso: Dictionary
        A WALTzER TSO data, output from running waltz (stage 1, see example).
    depth_model: 2D float array
        Transit depth model, array of shape [2,nwave] with:
        - wavelength (micron)
        - transit depth
        Ignore for stare mode.
    obs_type: string
        The observing geometry 'transit' or 'eclipse', or 'stare'.
    n_obs: Integer
        Number of observations to co-add.
    transit_dur: Float
        In-transit time (hours).
        Ignore for stare mode.
    obs_dur: Float
        Total observation duration (hours).
    binsize: Integer
        Binning over spectral axis, in number of pixels.
    resolution: Float
        Alternative to binsize, set output resolution.
    noiseless: Bool
        For stare obs_type, set to True to prevent adding scatter
        the simulated spectrum.
    efficiency: Float
        WALTzER duty cycle efficiency. If None, taken from tso.

    Returns
    -------
    For transit or eclipse obs_type:
        bands: 1D string list
            WALTzER band names.
        bin_wl: list of 1D float arrays
            Binned wavelength arrays for each band (micron).
        bin_in: list of 1D float arrays
            Binned in-transit collected fluxes (number of electrons).
        bin_out: list of 1D float arrays
            Binned out-of-transit collected fluxes (number of electrons).
        bin_var_in: list of 1D float arrays
            Binned in-transit variances.
        bin_var_out: list of 1D float arrays
            Binned out_of-transit variances.
        bin_widths: list of 1D float arrays
            Bin-widths for each data point.
        bin_depth: list of 1D float arrays
            Binned transit/eclipse depths for each band.

    For stare obs_type:
        bands: 1D string list
            WALTzER band names.
        bin_wl: list of 1D float arrays
            Binned wavelength arrays for each band (micron).
        bin_flux: list of 1D float arrays
            Binned collected fluxes (number of electrons).
            (the detector throughputs have been corrected-out from these).
        bin_var: list of 1D float arrays
            Binned-flux variances.
        bin_widths: list of 1D float arrays
            Bin-widths for each data point.
    """
    if obs_type == 'stare':
        transit_dur = 0.0
        depth_model = 0.0

    # An interpolator to eval the depth over the WALTzER bands
    if np.isscalar(depth_model):
        res = 6000.0
        wl_model = ps.constant_resolution_spectrum(0.23, 20.0, res)
        depth = np.tile(depth_model, len(wl_model))
    else:
        wl_model, depth = depth_model
    model = si.interp1d(
        wl_model, depth, kind='slinear',
        bounds_error=False, fill_value='extrapolate',
    )

    if efficiency is None:
        efficiency = tso['meta']['efficiency']
    total_time = (obs_dur * 3600) * efficiency * n_obs
    dt_in = (transit_dur * 3600) * efficiency * n_obs
    dt_out = total_time - dt_in

    bands = tso['meta']['bands']
    data = []
    for band in bands:
        det = tso[band]
        wl = det['wl']
        flux = det['flux']
        var = det['variance']
        throughput = det['throughput']
        half_width = det['half_widths']

        bin_data = bin_tso_data(
            wl, flux, var, throughput, half_width,
            model, obs_type, dt_in, dt_out,
            det['det_type'], binsize, resolution,
        )

        if obs_type == 'stare':
            # Undo throughput scaling
            if det['det_type'] == 'photometry':
                det_wl = det['wl_min'], det['wl_max']
                det_resp = np.tile(det['throughput'], 2)
            else:
                det_wl = wl
                det_resp = det['throughput']
            throughput = si.interp1d(
                det_wl, det_resp, kind='slinear',
                bounds_error=False, fill_value='extrapolate',
            )
            response = throughput(wl)
            bin_out = bin_data[2] / response
            bin_var = bin_data[4] / response**2.0
            if not noiseless:
                np.random.seed(random.randint(0, 100000))
                rand_noise = np.random.normal(0.0, np.sqrt(bin_var))
                bin_out += rand_noise
            bin_data = (
                bin_data[0],
                bin_data[1], bin_out,
                bin_data[3], bin_var,
                bin_data[5], bin_data[6],
            )

        data.append(bin_data)


    bin_wl = [d[0] for d in data]
    bin_in = [d[1] for d in data]
    bin_out = [d[2] for d in data]
    bin_var_in = [d[3] for d in data]
    bin_var_out = [d[4] for d in data]
    bin_widths = [d[5] for d in data]
    bin_depth = [d[6] for d in data]

    if obs_type == 'stare':
        return (
            bands,
            bin_wl,
            bin_out,
            bin_var_out,
            bin_widths,
        )

    return (
        bands,
        bin_wl,
        bin_in,
        bin_out,
        bin_var_in,
        bin_var_out,
        bin_widths,
        bin_depth,
    )


def simulate_spectrum(
        tso, depth_model, obs_type='transit', n_obs=1,
        transit_dur=None, obs_dur=None,
        binsize=None, resolution=None, noiseless=False,
        efficiency=None,
    ):
    """
    Simulate a WALTzER TSO observation by combining SNR data with a
    transmission spectrum.

    Parameters
    ----------
    tso: Dictionary
        A WALTzER TSO data, output from running waltz (stage 1, see example).
    depth_model: 2D float array
        Transit depth model, array of shape [2,nwave] with:
        - wavelength (micron)
        - transit depth
        (sample at R > ~10000 for best results)
    obs_type: string
        The observing geometry 'transit' or 'eclipse'.
    n_obs: Integer
        Number of transits to co-add.
    transit_dur: Float
        If not None, overwrite transit duration (h) from tso dictionary.
    obs_dur: Float
        Total observation duration (h).
        If None, assume an out-of-transit duration of 2*max(t_dur, 1h).
    binsize: Integer
        Bin-down output spectrum by given bin size.
        Output resolution will be R = 6000.0 / binsize
        Note that 'resolution' input take precedence over binsize.
    resolution: Float
        Bin-down output spectrum by given resolution.
        Note this input take precedence over binsize.
    noiseless: Bool
        Set to True to prevent adding scatter the simulated spectrum.
    efficiency: Float
        WALTzER duty-cycle efficiency.

    Returns
    -------
    bands: 1D string list
        The WALTzER band names.
    walz_wl: List of 1D float arrays
        WALTzER wavelength array for NUV, VIS, and NIR bands (micron)
    walz_spec: List of 1D float arrays
        WALTzER transit depths for each band
    walz_err: List of 1D float arrays
        WALTzER transit-depth uncertainties for each band
    walz_widths: List of 1D float arrays
        Wavelength half-widths of WALTzER data points (micron).

    Examples
    --------
    >>> import pickle
    >>> import numpy as np
    >>> import pyratbay.constants as pc
    >>> import waltzer_etc as w

    >>> # Load a WALTzER SNR output pickle file
    >>> tso_file = 'waltzer_snr.pickle'
    >>> with open(tso_file, 'rb') as handle:
    >>>     spectra = pickle.load(handle)
    >>> tso = spectra['HD 209458 b']

    >>> # Load a transit-depth spectrum
    >>> tdepth_file = 'transit_saturn_1600K_clear.dat'
    >>> wl, depth = np.loadtxt(tdepth_file, unpack=True)
    >>> depth_model = wl, depth

    >>> # Simulate WALTzER observation
    >>> sim = w.simulate_spectrum(
    >>>     tso, depth_model,
    >>>     n_obs=10,
    >>>     resolution=300.0,
    >>>     noiseless=False,
    >>> )

    >>> # Show noised-up WALTzER spectrum
    >>> bands, waltzer_wl, waltzer_spec, waltzer_err, waltzer_widths = sim
    >>> fig = plt.figure(1)
    >>> plt.clf()
    >>> fig.set_size_inches(8,5)
    >>> plt.subplots_adjust(0.1,0.1,0.98,0.98, hspace=0.15)
    >>> ax = plt.subplot(3,1,(1,2))
    >>> plt.plot(depth_model[0], depth_model[1]/pc.percent, color='xkcd:blue')
    >>> bands = tso['meta']['bands']
    >>> for j,band in enumerate(bands):
    >>>     plt.errorbar(
    >>>         waltzer_wl[j], waltzer_spec[j]/pc.percent,
    >>>         waltzer_err[j]/pc.percent, xerr=waltzer_widths[j],
    >>>         fmt='o', ecolor='salmon', color='xkcd:orangered',
    >>>         mfc='w', ms=4, zorder=0,
    >>>     )
    >>> plt.xscale('log')
    >>> ax.set_xticks([0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6])
    >>> ax.set_xticklabels([])
    >>> ax.tick_params(which='both', direction='in')
    >>> plt.xlim(0.24, 1.7)
    >>> plt.ylim(0.99, 1.12)
    >>> ax.set_ylabel('Transit depth (%)')

    >>> ax = plt.subplot(3,1,3)
    >>> for j,band in enumerate(bands):
    >>>     ax.errorbar(
    >>>         waltzer_wl[j], waltzer_err[j]/pc.ppm, xerr=waltzer_widths[j],
    >>>         fmt='o', ecolor='salmon', color='tomato',
    >>>         mfc='w', ms=4, zorder=0,
    >>>     )
    >>> ax.set_xscale('log')
    >>> ax.set_yscale('log')
    >>> ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    >>> ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    >>> ax.set_xticks([0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.6])
    >>> ax.set_xlim(0.24, 1.7)
    >>> ax.tick_params(which='both', direction='in')
    >>> ax.set_xlabel('Wavelength (um)')
    >>> ax.set_ylabel('Depth error (ppm)')
    """
    # An interpolator to eval the depth over the WALTzER bands
    if np.isscalar(depth_model):
        wl_model = ps.constant_resolution_spectrum(0.23, 20.0, resolution=6000)
        depth = np.tile(depth_model, len(wl_model))
        fill_value = 'extrapolate'
    else:
        wl_model, depth = depth_model
        fill_value = np.nan

    model = si.interp1d(
        wl_model, depth, kind='slinear',
        bounds_error=False, fill_value=fill_value,
    )

    if efficiency is None:
        efficiency = tso['meta']['efficiency']
    if n_obs is None:
        n_obs = tso['meta']['n_obs']

    if transit_dur is None:
        transit_dur = tso['meta']['transit_dur']
    if obs_dur is None:
        obs_dur = transit_dur + 2.0*np.amax([1.5*transit_dur, 1.0])

    # Total times integrating in- and out-of-transit (in seconds)
    total_time = (obs_dur * 3600) * efficiency * n_obs
    dt_in = (transit_dur * 3600) * efficiency * n_obs
    dt_out = total_time - dt_in

    bands = tso['meta']['bands']

    walz_wl = []
    walz_depth = []
    walz_err = []
    walz_widths = []
    for band in bands:
        det = tso[band]
        wl = det['wl']
        flux = det['flux']
        var = det['variance']
        throughput = det['throughput']
        half_width = det['half_widths']

        bin_data = bin_tso_data(
            wl, flux, var, throughput, half_width,
            model, obs_type, dt_in, dt_out,
            det['det_type'], binsize, resolution,
        )

        bin_wl = bin_data[0]
        bin_in = bin_data[1]
        bin_out = bin_data[2]
        bin_var_in = bin_data[3]
        bin_var_out = bin_data[4]
        bin_widths = bin_data[5]
        bin_depth = bin_data[6]
        # Error propagation into transit-depth uncertainty
        bin_err = np.sqrt(
            (dt_out / dt_in / bin_out)**2.0 * bin_var_in +
            (bin_in / dt_in * dt_out / bin_out**2.0)**2.0 * bin_var_out
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
        walz_depth.append(bin_spec)
        walz_err.append(bin_err)
        walz_widths.append(bin_widths)

    return bands, walz_wl, walz_depth, walz_err, walz_widths



