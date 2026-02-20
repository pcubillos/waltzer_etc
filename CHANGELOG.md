# CHANGELOG

## Version 0.3.4 (2026-02-20)

- Refactored code to perform more accurate transit-depth SNRs, performing calculations at high-resolution, then convolving to WALTzER resolving power, and lastly integrating over WALTzER pixels.
- Implemented '--hires' argument to command line call, allowing to set the internal high-resolution sampling.  Default (and minimum accepted value) is 48_000.
- Removed simulate_fluxes() function, now incorporated into simulate_spectrum().

## Version 0.3.3 (2026-02-12)

- Updated throughputs

## Version 0.3.2 (2026-02-10)

- Enabled '--obs_mode' argument to command line call, which allows to set 'stare' mode (i.e., no transits).
- Enabled custom SED inputs in command line call, by adding a column with header 'sed', for each target, users can set the path to an SED file.
- simulate_fluxes() can now produce simulated source spectra for stare mode.
- Updated primary-mirror size assumed in command line call to 35cm.

## Version 0.3.1 (2026-01-28)

- Enabled blackbody planet spectrum for eclipse geometry.
- Enabled 'Save data' button to save SED model when the display tab is at "Stellar SED"

## Version 0.3.0 (2026-01-26)

Updated instrument according to step1-revision

- Increased primary diameter from 30 to 35 cm
- NUV band pass shifted from 0.25-0.33um to 0.24-0.32um to capture additional Fe bands
- The `.csv` output from command-line call now includes the minimum fluxes and SNRs in band
- The `.csv` output from command-line call replaced mean fluxes and SNRs in band with the median
- Added quantum-efficiency throughputs as function of wavelength 
- Added PHOENIX and BT-Settl stellar SEDs
- Added option to upload custom stellar SEDs
- Noise and TSO resolution inputs now show the true resolution being used (data is binned down by # of pixels: 2, 3, 4, ...etc. So there's only a fixed number of resolutions possible, particularly at the highest resolutions).
- Updated JWST-programs catalog (there were some missing programs before)

