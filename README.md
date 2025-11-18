# :rocket: WALTzER ETC :telescope:
ETC for the WALTzER F-mission concept

## Instructions

### Install code

Download the code with this shell command:
```shell
git clone https://github.com/pcubillos/waltzer_etc
```

### Install requirements

Go into the ``waltzer_etc`` and execute this shell commad to install the required Python packages
```shell
pip install -r requirements.txt
```

### Download auxiliary files

Fetch stellar SED's from this link: https://drive.google.com/file/d/1Xlg6qLCSNAlZ9gXwupWl-58bkfJ-6dhR/view?usp=sharing

Unzip the file and place the ``models`` folder in the same directory where snr_waltzer.py is located.

## Stage 1: run the SNR

You will need an input .csv file with the targets to simulate (e.g., from a NASA Exoplanet Archive request). This file must contain these headers:
- `'pl_name'`
- `'pl_trandur'`
- `'st_teff'`
- `'st_rad'`
- `'st_mass'`
- `'ra'`
- `'dec'`
- `'sy_dist'`
- `'sy_vmag'`

There is an example target list ``'target_list_20250327.csv'`` in the repository.

Run the code with this shell command:
```shell
python snr_waltzer.py target_list.csv waltzer_snr.csv
```

This will produce an output .csv file ``waltzer_snr.csv`` with
- mean and max fluxes at each band (NUV, VIS, and NIR)
- mean and max SNR at each band
- mean transit-depth uncertainties at each band

Spectroscopic SNR values are per sampling element, i.e., per each HWHM,
given the WALTzER's resolution of R = lambda/FWHM = 3000.0
Photometric SNR values (NIR) are band-integrated.


### Extra

Run this command to see all options
```shell
python snr_waltzer.py -h
```


Run this command to compute stats for a fixed time duration (h)
```shell
python snr_waltzer.py target_list_20250327.csv waltzer_snr.csv --tdur=2.5
```


## Stage 2: simulate transit depth spectra

Stage 1 will also produce a pickle file with the noise estimation for
each source.  Combined this with a transit-depth model to simulate
WALTzER observations (a sample transit-depth file is provided in the repo):

```python
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyratbay.constants as pc
import snr_waltzer as w
plt.ion()

# Load WALTzER SNR output pickle file and select a target
tso_file = 'waltzer_snr_test.pickle'
with open(tso_file, 'rb') as handle:
    spectra = pickle.load(handle)
tso = spectra['HD 209458 b']

# Load a transit-depth spectrum
tdepth_file = 'transit_saturn_1600K_clear.dat'
depth_model = np.loadtxt(tdepth_file, unpack=True)

sim = w.simulate_spectrum(
    tso, depth_model,
    n_obs=10,
    resolution=300.0,
    noiseless=False,
)


# Plot model and WALTzER simulation
waltzer_wl, waltzer_spec, waltzer_err, waltzer_widths = sim

fs = 12
fig = plt.figure(0)
plt.clf()
plt.subplots_adjust(0.09, 0.12, 0.98, 0.98, hspace=0.18)
fig.set_size_inches(8,4)
ax = plt.subplot(111)
ax.plot(depth_model[0], depth_model[1]/pc.percent, color='xkcd:blue')
bands = ['NUV', 'VIS', 'NIR']
for j,band in enumerate(bands):
    plt.errorbar(
        waltzer_wl[j], waltzer_spec[j]/pc.percent,
        waltzer_err[j]/pc.percent, xerr=waltzer_widths[j],
        fmt='o', ecolor='cornflowerblue', color='royalblue',
        mfc='w', ms=4, zorder=0,
    )
ax.set_xscale('log')
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks([2500, 3000, 4000, 5000, 7000, 10000, 16000])
ax.set_xlim(2400, 17000)
ax.set_ylim(0.99, 1.12)
ax.set_xlabel('Wavelength (um)', fontsize=fs)
ax.set_ylabel('Depth error (ppm)', fontsize=fs)
ax.tick_params(which='both', right=True, direction='in', labelsize=fs-1)
```
