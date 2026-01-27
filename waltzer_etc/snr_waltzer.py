# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'Detector',
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

from .utils import ROOT


DEFAULT_DETS = {
    'nuv': f'{ROOT}/data/detectors/waltzer_nuv.cfg',
    'vis': f'{ROOT}/data/detectors/waltzer_vis.cfg',
    'nir': f'{ROOT}/data/detectors/waltzer_nir.cfg',
}


def calc_collecting_area(diameter, band):
    """
    Calculate the effective collecting area of the telescope+detector
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
    Uv_detQE = 0.55  # UV detector QE in %

    R_opfold = 0.90  # Optical fold reflectance in %
    R_opgr   = 0.90  # Optical grating reflectance in %
    Op_geff  = 0.75  # Optical grating effeciency in %
    Op_detQE = 0.80  # Optical detector QE in %

    BB_detQE = 0.80  # IR Broad band detector QE in %

    # Effective collecing areas in cm^2
    if band == 'nuv':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_uvfold * R_uvgr * Uv_geff
            #* Uv_detQE
        )
        return eff_area

    if band == 'vis':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_d2 * R_opfold**2 * R_opgr * Op_geff
            #* Op_detQE
        )
        return eff_area

    if band == 'nir':
        eff_area = (
            primary_area * Rprim * Rsec * Sec_obstr
            * R_d1 * R_d2 * R_opfold
            #* BB_detQE
        )
        return eff_area

    raise ValueError(f'Invalid band {repr(band)}')


def throughput(file):
    """
    Load a quantum-efficiency curve into a scipy interpolator function
    if input file is a number, assume that as fixed QE.
    """
    if os.path.exists(file):
        wl_det, response = np.loadtxt(file, unpack=True)
    else:
        resp = float(os.path.split(file)[1])
        wl_det = [0.2, 2.0]
        response = np.tile(resp, 2)

    response /= 100.0
    fill_value = response[0], response[-1]
    throughput = si.interp1d(
        wl_det, response,
        kind='slinear', bounds_error=False, fill_value=fill_value,
    )
    return throughput


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

        self.eff_area = calc_collecting_area(diameter, self.band)
        qe_file = f'{ROOT}/data/detectors/{det.get("qe_file")}'
        self.throughput = throughput(qe_file)

        self.resolution = det.getfloat('resolution')
        self.pix_scale = det.getfloat('pix_scale')
        self.dark = det.getfloat('dark')
        self.read_noise = det.getfloat('read_noise')
        self.exp_time = det.getfloat('exp_time')
        self.aperture = det.getint('aperture')

        self.wl_min = det.getfloat('wl_min')
        self.wl_max = det.getfloat('wl_max')

        wl_edges = ps.constant_resolution_spectrum(
            0.24, 2.0, resolution=2.0*self.resolution,
        )
        i_min = np.searchsorted(wl_edges, self.wl_min)
        i_max = np.searchsorted(wl_edges, self.wl_max)
        self._wl_edges = wl_edges[i_min:i_max]
        # Center and half-widths of wavelength bins (micron):
        self.wl = 0.5 * (wl_edges[i_min+1:i_max] + wl_edges[i_min:i_max-1])
        self.half_widths = self.wl - wl_edges[i_min:i_max-1]

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

    def photon_spectrum(self, wl, flux):
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
        throughput = self.throughput(wl)
        # Convert flux (mJy) to photons s-1 Hz-1
        photon_energy = pc.h * pc.c / (wl*pc.um)
        photons_nu = 1e-26 * flux / photon_energy * self.eff_area * throughput
        # Now convert to photons s-1 um-1
        photons = photons_nu * pc.c / (wl*pc.um)**2.0 * pc.um

        # Background flux (erg s-1 cm-2 A-1 pixel-1)
        bkg_flux = self.bkg_model(wl) * self.pix_scale**2.0
        # Convert flux (erg s-1 cm-2 A-1 pixel-1) to photons s-1 um-1 pixel-1
        bkg_photons = bkg_flux / photon_energy * self.eff_area * throughput * pc.um/pc.A

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
        edge_idx = np.searchsorted(wl, wl_edges)
        bin_flux = np.zeros(self.nwave)
        bin_bkg = np.zeros(self.nwave)
        for i in range(self.nwave):
            i1, i2 = edge_idx[i], edge_idx[i+1]+1
            bin_flux[i] = np.trapezoid(photons[i1:i2], wl[i1:i2])
            bin_bkg[i] = np.trapezoid(bkg_photons[i1:i2], wl[i1:i2])

        return bin_flux, bin_bkg


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
        band = (wl > self.wl_min) & (wl < self.wl_max)
        min_flux = np.min(flux[band])
        median_flux = np.median(flux[band])
        max_flux = np.max(flux[band])
        return min_flux, median_flux, max_flux


    def calc_total_noise(self, wl, flux, integ_time=1.0):
        """
        Compute the time-integrated source flux and total variance spectra.

        Parameters
        ----------
        wl: 1D float array
            Input wavelength array (micron).
        flux: 1D float array
            Input flux spectrum (mJy).
        integ_time: Float
            Total integration time (s).  Leave as integ_time=1.0 to
            obtain the values per second.

        Returns
        -------
        source: float
            Source's time-integrated flux spectrum in number of photons.
        variance: float
            Observation's variance in number of photons.
        """
        variances = self.calc_noise(wl, flux, integ_time)

        source = variances[0]
        variance = np.sum(np.array(variances), axis=0)
        return source, variance


    def calc_noise(self, wl, flux, integ_time=1.0):
        """
        Compute the time-integrated components of the noise:
        source, background, dark, and read noise.

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
        var_source: 1D float array
            Source Poisson noise variance.
        var_background: 1D float array
            Background Poisson noise variance.
        var_dark: 1D float array
            Dark-current variance.
        var_read: 1D float array
            Read-noise variance.
        """
        if integ_time is None:
            integ_time = 1.0
        # For the time being, ignore the fact that nreads should be integer
        nreads = integ_time / self.exp_time

        # Fluxes in photons per second
        e_flux, e_background = self.photon_spectrum(wl, flux)

        # integrate over time
        total_flux = e_flux * integ_time
        var_source = np.abs(total_flux)

        # Aperture sizes
        aperture_radius = self.aperture//2

        # Number of pixels in source and in sky-background
        if self.mode == 'photometry':
            npix = int(np.pi*aperture_radius**2)
            sky_in = aperture_radius
            sky_out = 2*aperture_radius
            nsky = int(np.pi * (sky_out**2 - sky_in**2))
        elif self.mode == 'spectroscopy':
            npix = 2*aperture_radius
            sky_in = aperture_radius
            sky_out = 3*aperture_radius
            nsky = 2 * (sky_out-sky_in)
        else:
            raise ValueError()

        # Background number of photons
        var_background = npix*(1+npix/nsky) * e_background * integ_time

        # Dark number of photons
        var_dark = npix*(1+npix/nsky) * self.dark * integ_time
        var_dark = np.tile(var_dark, self.nwave)

        # Total number of photons per pixel
        var_read = npix*(1+npix/nsky) * self.read_noise * nreads
        var_read = np.tile(var_read, self.nwave)

        return (
            var_source,
            var_background,
            var_dark,
            var_read,
        )


    def snr_stats(self, wl, flux, integ_time):
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
        # integrate over time
        total_flux, variance = self.calc_total_noise(wl, flux, integ_time)

        # Signal-to-noise estimation
        snr = total_flux / np.sqrt(variance)
        snr_min = np.min(snr)
        snr_median = np.median(snr)
        snr_max = np.max(snr)

        # Assume t_in = t_out
        # Assume flux_in approx flux_out
        transit_uncert = np.sqrt(2.0) / snr_median / pc.ppm

        return snr_min, snr_median, snr_max, transit_uncert


def bin_tso_data(
        wl, flux, var, half_width,
        depth_model, obs_type, dt_in, dt_out,
        det_type, binsize=None, resolution=None,
    ):
    """
    Combine a WALTzER output SNR data with a transmission spectrum
    to simulate observations.

    Parameters
    ----------
    tso: Dictionary
        A planet's model, output from running waltz (stage 1).
    depth_model: 2D float array
        Transit depth model, array of shape [2,nwave] with:
        - wavelength (micron)
        - transit depth
    n_obs: Integer
        Number of transits to co-add.
    resolution: Float
        Output resolution.
    transit_dur: Float
        If not None, overwrite transit duration (h) from tso dictionary.
    obs_type: string
        The observing geometry 'transit' or 'eclipse'.
    efficiency: Float
        WALTzER duty cycle efficiency.
    noiseless: Bool
        Set to True to prevent adding scatter the simulated spectrum.
    """
    # Evaluate transit-depth in the band:
    if det_type == 'photometry':
        band = ps.Tophat(wl[0], half_width[0], wl=depth_model.x, ignore_gaps=True)
        if band.wl is None:
            depth = np.nan
        else:
            depth = band.integrate(depth_model.y)
    else:
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
        tso, depth_model, obs_type='transit', n_obs=1,
        transit_dur=None, obs_dur=None,
        binsize=None, resolution=None, noiseless=False,
        efficiency=None,
    ):
    """
    Simulate WALTzER TSO fluxes and variances during an observation.
    """
    # An interpolator to eval the depth over the WALTzER bands
    if np.isscalar(depth_model):
        wl_model = ps.constant_resolution_spectrum(0.245, 20.0, resolution=6000)
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
        half_width = det['half_widths']

        bin_data = bin_tso_data(
            wl, flux, var, half_width,
            model, obs_type, dt_in, dt_out,
            det['det_type'], binsize, resolution,
        )
        data.append(bin_data)

    bin_wl = [d[0] for d in data]
    bin_in = [d[1] for d in data]
    bin_out = [d[2] for d in data]
    bin_var_in = [d[3] for d in data]
    bin_var_out = [d[4] for d in data]
    bin_widths = [d[5] for d in data]
    bin_depth = [d[6] for d in data]
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
        wl_model = ps.constant_resolution_spectrum(0.24, 20.0, resolution=6000)
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
        obs_dur = transit_dur + 2.0*np.amax([0.5*transit_dur, 1.0])

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
        half_width = det['half_widths']

        bin_data = bin_tso_data(
            wl, flux, var, half_width,
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
            (dt_out/dt_in/bin_out)**2.0 * bin_var_in +
            (bin_in/dt_in*dt_out/bin_out**2.0)**2.0 * bin_var_out
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



