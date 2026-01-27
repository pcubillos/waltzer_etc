# Copyright (c) 2025-2026 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

import json
import os
from pathlib import Path
from datetime import timedelta, datetime
from functools import reduce
import operator
import pickle

import faicons as fa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import ui, render, reactive, req, App
from shinywidgets import output_widget, render_plotly

import pyratbay.constants as pc
import pyratbay.spectrum as ps

from waltzer_etc.utils import ROOT as ROOT
from waltzer_etc.utils import inst_convolution
import waltzer_etc as waltz
import waltzer_etc.sed as sed
from gui_utils import (
    make_sed_catalog,
    get_throughput,
    get_auto_sed,
    planet_model_name,
    read_depth_spectrum,
    make_tso_label,
    parse_obs,
    parse_sed,
    _safe_num,
    read_spectrum_file,
    pretty_print_target,
    as_str,
    esasky_js_circle,
    esasky_js_catalog,
    custom_card,
    Catalog,
    fetch_gaia_targets,
)
# TBD proper import local file
from gui_popovers import (
    depth_units,
    wl_scales,
    flux_scales,
    noise_choices,
    tso_choices,
)

import gui_plotly as plt

bins = np.arange(6000, 0, -1)
resolutions = 6000.0 / bins


def searchsorted_closest(array, val):
    """
    Find index closest to val in a sorted array
    """
    if val <= array[0]:
        return 0
    if val >= array[-1]:
        return len(array) - 1

    idx = np.searchsorted(array[:-1], val)
    d_lo = val - array[idx-1]
    d_hi = array[idx] - val

    if d_lo < d_hi:
        return idx -1
    return idx


def quick_snr(tso, t_in, t_out=None, depth=0.0):
    # Total times integrating in- and out-of-transit
    bands = list(tso)

    snr = []
    t_error = []
    for j,band in enumerate(bands):
        sim = tso[band]
        flux = sim['variance'][0]
        var = np.sum(np.array(sim['variance']), axis=0)

        # Transit depth and fluxes [e- per second]
        #depth = model(wl)
        flux_out = flux
        flux_in = flux * (1.0-depth)
        # Noise estimation for total number of e- collected
        var_out = var * t_out
        var_in = var * (1.0-depth) * t_in

        # Error propagate to into transit-depth uncertainty
        transit_depth_error = np.sqrt(
            (1.0/t_in/flux_out)**2.0 * var_in +
            (flux_in/t_out/flux_out**2.0)**2.0 * var_out
        )
        t_error.append(transit_depth_error/pc.ppm)

        # Time-integrated SNR on the stellar flux
        signal = flux_in*t_in + flux_out*t_out
        snr.append(signal/np.sqrt(var_out))

    return snr, t_error


# The three amigos
bands = ['nuv', 'vis', 'nir']
detectors = {
    band: waltz.Detector(band)
    for band in bands
}

band_wl, band_response, band_mask = get_throughput()

def masked_throughput(masks):
    if len(masks) == 0:
        masks = list(band_mask)
    mask = reduce(operator.or_, (band_mask[k] for k in masks))
    throughput = {
        'wl': band_wl,
        'response': band_response*mask,
    }
    return throughput



def load_catalog():
    catalog = Catalog()
    is_jwst = np.array([target.is_jwst_planet for target in catalog.targets])
    is_transit = np.array([target.is_transiting for target in catalog.targets])
    is_confirmed = np.array([target.is_confirmed for target in catalog.targets])
    return catalog, is_jwst, is_transit, is_confirmed


# Catalog of known exoplanets (and candidate planets)
catalog, is_jwst, is_transit, is_confirmed = load_catalog()
nplanets = len(catalog.targets)

# Catalog of stellar SEDs:
sed_choices = {
    "llmodels": "llmodels",
    "phoenix": "phoenix",
    #"k93models": "kurucz (k93models)",
    "bt_settl": "BT-Settl MLT (bt_settl)",
    #"blackbody": "blackbody",
    "input": "input",
}
sed_catalog, sed_dict = make_sed_catalog()


# Higher resolution for models (will be bin down to WALTzER)
resolution = 45_000.0
wl = ps.constant_resolution_spectrum(0.23, 2.0, resolution=resolution)
# WALTzER's resolution
inst_resolution = detectors['vis'].resolution
cache_seds = {}


def waltz_model(wl_model, depth):
    """
    Resample model and apply WALTzER instrumental resolving power.
    """
    # interpolate
    interp_depth = np.interp(
        wl, wl_model, depth, left=np.nan, right=np.nan,
    )
    waltzer_depth = inst_convolution(
        wl, interp_depth, inst_resolution, sampling_res=resolution,
        mode='valid',
    )
    edge = (wl.size - waltzer_depth.size) // 2
    waltzer_wl = wl[edge:-edge]
    return waltzer_wl, waltzer_depth



def load_sed(sed_model, sed_type, cache_seds):
    """
    load SED wrapper with cache'd files.
    """
    if sed_model in cache_seds:
        sed_flux = cache_seds[sed_model]
    else:
        if sed_type == 'input':
            sed_wl = spectra['sed'][sed_model]['wl']
            flux = spectra['sed'][sed_model]['flux']
        else:
            teff = sed_catalog[sed_model]['teff']
            logg = sed_catalog[sed_model]['logg']
            sed_type = sed_catalog[sed_model]['sed_type']
            sed_wl, flux = sed.load_sed(teff, logg, sed_type)

        flux = np.interp(wl, sed_wl, flux)
        sed_flux = inst_convolution(
            wl, flux, inst_resolution, sampling_res=resolution,
        )
        cache_seds[sed_model] = sed_flux
    return sed_flux


# SED normalization band
band_name = 'johnson,v'

depth_choices = {
    'transit': ['Flat', 'Input'],
    'eclipse': ['Blackbody', 'Input']
}

tso_runs = {
    'Transit': {},
    'Eclipse': {},
    'SNR': {},
}

def make_tso_labels(tso_runs):
    tso_labels = {
        'Transit': {},
        'Eclipse': {},
    }
    for key, runs in tso_runs.items():
        for tso_label, tso in runs.items():
            tso_key = f'{key}_{tso_label}'
            tso_labels[key][tso_key] = tso['meta']['label']
    return tso_labels


cache_target = {}
cache_acquisition = {}
cache_saturation = {}

# Planet and stellar spectra
spectra = {
    'transit': {},
    'eclipse': {},
    'sed': {},
}
bookmarked_spectra = {
    'transit': [],
    'eclipse': [],
    'sed': [],
}

# Load spectra from user-defined folder and/or from default folder
#loading_folders = []
#argv = [arg for arg in sys.argv if arg != '--debug']
#if len(argv) == 2:
#    loading_folders.append(os.path.realpath(argv[1]))
#loading_folders.append(f'{ROOT}data/models')
current_dir = os.path.realpath(os.getcwd())

#for location in loading_folders:
#    t_models, e_models, sed_models = collect_spectra(location)
#    for label, model in t_models.items():
#        spectra['transit'][label] = model
#    for label, model in e_models.items():
#        spectra['eclipse'][label] = model
#    for label, model in sed_models.items():
#        spectra['sed'][label] = model


nasa_url = 'https://exoplanetarchive.ipac.caltech.edu/overview'
stsci_url = 'https://www.stsci.edu/jwst/science-execution/program-information?id=PID'
cdnjs = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/'

# Depth and SED units, ensure they are consistent with read_spectrum_file()
wl_units = [
    "angstrom",
    "micron",
]
ergs_s_cm2 = "erg s\u207b\u00b9 cm\u207b\u00b2"
sed_units = {
    "f_freq": f"{ergs_s_cm2} Hz\u207b\u00b9 (frequency space)",
    "f_nu": f"{ergs_s_cm2} cm (wavenumber space)",
    "f_lambda": f"{ergs_s_cm2} cm\u207b\u00b9 (wavelength space)",
    "mJy": "mJy",
}

layout_kwargs = dict(
    width=1/2,
    fixed_width=False,
    heights_equal='all',
    gap='7px',
    fill=False,
    fillable=True,
    class_="pb-2 pt-0 m-0",
)

welcome = f"""\
To update:

- cd into the `waltzer_etc/` folder containing `pyproject.toml`
- then run these commands:
```shell
git pull origin main
pip install -e .
```
"""

card_style = "background:#F5F5F5; !important;"

app_ui = ui.page_fluid(
    # ESA Sky
    ui.tags.script(
        """
        $(function() {
            Shiny.addCustomMessageHandler("update_esasky", function(message) {
                var esaskyFrame = document.getElementById("esasky");
                esaskyFrame.contentWindow.postMessage(
                    JSON.parse(message.command), 'https://sky.esa.int'
                );
            });
        });
        """
    ),
    # NASA link
    ui.tags.script("""
        Shiny.addCustomMessageHandler("set_nasa_href", function(msg) {
            const link = document.getElementById("nasa_link");
            if (link) link.href = msg.url;
        });
    """),
    # Copy to clipboard
    ui.HTML("""
        <script>
        Shiny.addCustomMessageHandler('copy_to_clipboard', function(message) {
            navigator.clipboard.writeText(message)
                .then(function() {
                    Shiny.setInputValue("copy_status", "success", {priority: "event"});
                })
                .catch(function(err) {
                    Shiny.setInputValue("copy_status", "failure", {priority: "event"});
                });
        });
        </script>
    """),
    # Syntax highlighting (python)
    ui.HTML(
        f'<link rel="stylesheet" href="{cdnjs}styles/base16/one-light.min.css">'
        f'<script src="{cdnjs}highlight.min.js"></script>'
    ),
    ui.tags.style(
        """
        .popover {
            --bs-popover-max-width: 500px;
        }
        """
    ),
    ui.layout_columns(
        ui.span(
            ui.HTML(
                "<b>WALTzER's</b> Exoplanet time-series observations ETC  "
            ),
            ui.tooltip(
                ui.input_action_link(
                    id='main_settings',
                    label='',
                    icon=fa.icon_svg("gear", fill='black'),
                ),
                "settings",
                placement='bottom',
            ),
            style="font-size: 26px;",
        ),
        col_widths=(12),
        fixed_width=False,
        fill=False,
        fillable=True,
    ),
    ui.layout_columns(
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The target
        custom_card(
            ui.card_header("Target", class_="bg-primary"),
            ui.card(
                ui.card_body(
                    # a hidden section to hold switches for other conditionals
                    ui.panel_conditional(
                        'false',
                        ui.input_switch(
                            id="is_candidate",
                            label="candidate",
                            value=False,
                        ),
                        ui.input_switch(
                            id="has_sed_bookmarks",
                            label="has SED",
                            value=False,
                        ),
                        ui.input_switch(
                            id="has_depth_bookmarks",
                            label="has transits",
                            value=False,
                        ),
                    ),
                    ui.popover(
                        ui.span(
                            fa.icon_svg("gear"),
                            style="position:absolute; top: 5px; right: 7px;",
                        ),
                        ui.input_checkbox_group(
                            id="target_filter",
                            label='',
                            choices={
                                "transit": "transiting",
                                "jwst": "JWST targets",
                                "tess": "TESS candidates",
                                "non_transit": "non-transiting",
                            },
                            selected=['jwst', 'transit'],
                        ),
                        title='Filter targets',
                        placement="right",
                        id="targets_popover",
                    ),
                    ui.span(
                        ui.HTML('<b>Target</b> '),
                        ui.tooltip(
                            ui.input_action_link(
                                id='show_info',
                                label='',
                                icon=fa.icon_svg("circle-info", fill='cornflowerblue'),
                            ),
                            'System info',
                            id='target_info_tooltip',
                            placement='top',
                        ),
                        #url has to be set with javascript, output_ui does not render nicely, ui.input_action_link() does not open in server side.
                        ui.tooltip(
                            ui.tags.a(
                                fa.icon_svg("circle-info", fill='black'),
                                id='nasa_link',
                                href=f'{nasa_url}',
                                target="_blank",
                            ),
                            "Open target's NASA Exoplanet Archive",
                            id='nasa_tooltip',
                            placement='top',
                        ),
                        ui.tooltip(
                            ui.input_action_link(
                                id='show_observations',
                                label='',
                                icon=fa.icon_svg("circle-info", fill='gray'),
                            ),
                            'not a JWST target (yet)',
                            id='jwst_tooltip',
                            placement='top',
                        ),
                        ui.panel_conditional(
                            "input.is_candidate",
                            ui.tooltip(
                                fa.icon_svg("triangle-exclamation", fill='darkorange'),
                                ui.markdown("This is a *candidate* planet"),
                                placement='top',
                            ),
                        ),
                    ),
                    ui.input_selectize(
                        id='target',
                        label='',
                        choices=[target.planet for target in catalog.targets],
                        selected='HD 209458 b',
                        multiple=False,
                    ),
                    # SED properties
                    ui.layout_column_wrap(
                        # Row 1
                        ui.HTML("<p>T<sub>eff</sub> (K):</p>"),
                        ui.input_numeric("t_eff", "", value='1400.0'),
                        # Row 2
                        ui.p("log(g):"),
                        ui.input_numeric("log_g", "", value='4.5'),
                        # Row 3
                        ui.p("V magnitude:"),
                        ui.input_numeric(
                            id="magnitude",
                            label="",
                            value='10.0',
                        ),
                        width=1/2,
                        fixed_width=False,
                        heights_equal='all',
                        gap='7px',
                        fill=False,
                        fillable=True,
                    ),
                    ui.span(
                        ui.HTML('<b>Stellar SED</b> '),
                        # upload (hidden for now)
                        ui.tooltip(
                            ui.input_action_link(
                                id='upload_sed',
                                label='',
                                icon=fa.icon_svg("file-arrow-up", fill='black'),
                            ),
                            'Upload SED',
                            id='sed_up_tooltip',
                            placement='top',
                        ),
                        # bookmarks
                        ui.tooltip(
                            ui.input_action_link(
                                id='bookmark_sed',
                                label='',
                                icon=fa.icon_svg("star", style='regular', fill='black'),
                            ),
                            'Bookmark SED',
                            id='sed_book_tooltip',
                            placement='top',
                        ),
                        # clear
                        ui.panel_conditional(
                            "input.has_sed_bookmarks",
                            ui.tooltip(
                                ui.input_action_link(
                                    id='clear_sed_bookmarks',
                                    label='',
                                    icon=fa.icon_svg("circle-xmark", style='regular', fill='black'),
                                ),
                                'Clear all SED bookmarks',
                                id='sed_clear_tooltip',
                                placement='top',
                            ),
                        ),
                    ),
                    ui.input_select(
                        id="sed_type",
                        label='',
                        choices=sed_choices,
                        selected=list(sed_choices)[0],
                    ),
                    ui.input_select(
                        id="sed",
                        label="",
                        choices=sed_dict[list(sed_choices)[0]],
                    ),
                    fill=False,
                    style=card_style,
                    class_="px-2 pt-2 pb-0 m-0 gap-2",
                ),
                class_="p-0 pb-1 m-0",
                fill=False,
            ),
            # The planet
            ui.card(
                ui.card_body(
                    ui.popover(
                        ui.span(
                            fa.icon_svg("gear"),
                            style="position:absolute; top: 5px; right: 7px;",
                        ),
                        ui.markdown(
                            '*T*<sub>dur</sub> = '
                            '*T*<sub>settle</sub> + *T*<sub>base</sub> + '
                            '*T*<sub>tran</sub> + *T*<sub>base</sub>',
                        ),
                        ui.markdown(
                            'Start time window (*T*<sub>start</sub>): 1h',
                        ),
                        ui.input_numeric(
                            id="settling_time",
                            label=ui.markdown(
                                'Settling time (*T*<sub>settle</sub>, h):',
                            ),
                            value = 0.0,
                            step = 0.25,
                        ),
                        ui.input_numeric(
                            id="baseline_time",
                            label=ui.markdown(
                                'Baseline time (*T*<sub>base</sub>, t_dur):',
                            ),
                            value = 0.5,
                            step = 0.25,
                        ),
                        ui.input_numeric(
                            id="min_baseline_time",
                            label='Minimum baseline time (h):',
                            value = 1.0,
                            step = 0.25,
                        ),
                        title='Observation duration',
                        placement="right",
                        id="obs_popover",
                    ),
                    ui.markdown("**Observation**"),
                    ui.layout_column_wrap(
                        # Row 1
                        ui.p("Type:"),
                        ui.input_select(
                            id='obs_geometry',
                            label='',
                            choices={
                                'transit': 'Transit',
                                'eclipse': 'Eclipse',
                            }
                        ),
                        # Row 2
                        ui.output_text('transit_dur_label'),
                        ui.input_numeric("t_dur", "", value='2.0', step=0.1),
                        # Row 3
                        ui.p("Obs_dur (h):"),
                        ui.input_numeric("obs_dur", "", value='5.0', step=0.25),
                        width=1/2,
                        fixed_width=False,
                        heights_equal='all',
                        gap='7px',
                        fill=False,
                        fillable=True,
                    ),
                    ui.span(
                        ui.output_ui('depth_label'),
                        # upload
                        ui.tooltip(
                            ui.input_action_link(
                                id='upload_depth',
                                label='',
                                icon=fa.icon_svg("file-arrow-up", fill='black'),
                            ),
                            "Upload depth model",
                            id='depth_up_tooltip',
                            placement='top',
                        ),
                        # bookmarks
                        ui.tooltip(
                            ui.input_action_link(
                                id='bookmark_depth',
                                label='',
                                icon=fa.icon_svg("earth-americas", style='solid', fill='gray')
                            ),
                            'Bookmark depth model',
                            id='depth_book_tooltip',
                            placement='top',
                        ),
                        # clear
                        ui.panel_conditional(
                            "input.has_depth_bookmarks",
                            ui.tooltip(
                                ui.input_action_link(
                                    id='clear_depth_bookmarks',
                                    label='',
                                    icon=fa.icon_svg("circle-xmark", style='regular', fill='black'),
                                ),
                                'Clear all depth bookmarks',
                                id='depth_clear_tooltip',
                                placement='top',
                            ),
                        ),
                    ),
                    ui.input_select(
                        id="planet_model_type",
                        label="",
                        choices=["Input"],
                    ),
                    ui.panel_conditional(
                        "input.planet_model_type == 'Input'",
                        ui.tooltip(
                            ui.input_select(
                                id="depth",
                                label="",
                                choices=list(spectra['transit']),
                            ),
                            '',
                            id='depth_tooltip',
                            placement='right',
                        ),
                    ),
                    ui.panel_conditional(
                        "input.planet_model_type == 'Flat'",
                        ui.layout_column_wrap(
                            ui.p("Depth (%):"),
                            ui.input_numeric(
                                id="transit_depth",
                                label="",
                                value=0.5,
                                step=0.1,
                            ),
                            **layout_kwargs,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.planet_model_type == 'Blackbody'",
                        ui.layout_column_wrap(
                            ui.HTML("<p>(Rp/Rs)<sup>2</sup> (%):</p>"),
                            ui.input_numeric(
                                id="eclipse_depth",
                                label="",
                                value=0.05,
                                step=0.1,
                            ),
                            ui.p("Temp (K):"),
                            ui.input_numeric(
                                id="teq_planet",
                                label="",
                                value=2000.0,
                                step=100,
                            ),
                            **layout_kwargs,
                        ),
                    ),
                    fill=False,
                    style=card_style,
                    class_="px-2 pt-2 pb-0 m-0 gap-2",
                ),
                class_="p-0 m-0",
                fill=False,
            ),
            body_args=dict(class_="p-2 m-0"),
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # The instrument setup
        custom_card(
            ui.card_header(
                'Instrument',
                class_="bg-primary",
            ),
            # filter
            ui.card(
                ui.card_body(
                    ui.input_checkbox_group(
                        id="bands",
                        label=ui.markdown('**Bands**'),
                        choices = {
                            "nuv": "NUV (0.24-0.32 um)",
                            "vis": "VIS (0.42-0.80 um)",
                            "nir": "NIR (0.90-1.60 um)",
                        },
                        selected=['nuv', 'vis', 'nir'],
                    ),
                    ui.input_numeric(
                        id="efficiency",
                        label="Efficiency (%)",
                        value=60.0,
                        min=0.0,
                        max=100.0,
                        step=0.5,
                    ),
                    ui.input_numeric(
                        id="n_obs",
                        label="Number of observations",
                        value=3,
                        min=1,
                        max=50,
                        step=1,
                    ),
                    style=card_style,
                    class_="px-2 pt-1 pb-2 m-0 gap-3",
                    fill=False,
                ),
                fill=False,
                class_="p-0 m-0",
            ),
            # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            # The plots setup
            ui.card(
                ui.card_body(
                    ui.markdown("**Plots**"),
                    ui.panel_conditional(
                        "input.tab === 'Stellar SED'",
                        ui.layout_column_wrap(
                            'Resolution:',
                            ui.input_numeric(
                                id='plot_sed_resolution',
                                label='',
                                value=0.0,
                                min=0.0, max=6000.0, step=50.0,
                            ),
                            "Flux scale:",
                            ui.input_select(
                                id="plot_sed_yscale",
                                label="",
                                choices=flux_scales,
                                selected='linear',
                            ),
                            width=1/2,
                            fixed_width=False,
                            heights_equal='all',
                            gap='5px',
                            fill=False,
                            fillable=True,
                            class_="p-0 pb-1 m-0",

                        ),
                        "Wavelength:",
                        ui.layout_column_wrap(
                            ui.input_numeric(
                                id='sed_wl_min', label='',
                                value=0.23, min=0.2, max=30.0, step=0.02,
                            ),
                            ui.input_numeric(
                                id='sed_wl_max', label='',
                                value=2.0, min=0.2, max=30.0, step=0.1,
                            ),
                            ui.input_select(
                                id='plot_sed_xscale',
                                label='',
                                choices=wl_scales,
                                selected='log',
                            ),
                            width=1/3,
                            fixed_width=False,
                            heights_equal='all',
                            gap='5px',
                            fill=False,
                            fillable=True,
                            class_="p-0 pb-2 m-0",
                        ),
                        class_="p-0 m-0",
                    ),

                    ui.panel_conditional(
                        "input.tab == 'transit_tab'",
                        ui.layout_column_wrap(
                            'Resolution:',
                            ui.input_numeric(
                                id='depth_resolution',
                                label='',
                                value=250.0,
                                min=0.0, max=6000.0, step=50.0,
                            ),
                            "Depth units:",
                            ui.input_select(
                                id="plot_depth_units",
                                label="",
                                choices=depth_units,
                                selected='percent',
                            ),
                            width=1/2,
                            fixed_width=False,
                            gap='5px',
                            fill=False,
                            fillable=True,
                            class_="p-0 pb-1 m-0",
                        ),
                        "Wavelength:",
                        ui.layout_column_wrap(
                            ui.input_numeric(
                                id='depth_wl_min', label='',
                                value=0.23, min=0.2, max=30.0, step=0.02,
                            ),
                            ui.input_numeric(
                                id='depth_wl_max', label='',
                                value=2.0, min=0.2, max=30.0, step=0.1,
                            ),
                            ui.input_select(
                                "plot_depth_xscale",
                                label="",
                                choices=wl_scales,
                                selected='log',
                            ),
                            width=1/3,
                            fixed_width=False,
                            gap='5px',
                            fill=False,
                            fillable=True,
                            class_="p-0 pb-2 m-0",
                        ),
                        class_="px-0 py-0 m-0",
                    ),

                    ui.panel_conditional(
                        "input.tab === 'Noise' || input.tab === 'TSO'",
                        ui.layout_column_wrap(
                            "Display:",
                            ui.div(
                                ui.panel_conditional(
                                    "input.tab === 'Noise'",
                                    ui.input_select(
                                        id="noise_plot",
                                        label="",
                                        choices=noise_choices,
                                        selected='variance',
                                    ),
                                    class_="px-0 py-0 m-0",
                                ),
                                ui.panel_conditional(
                                    "input.tab === 'TSO'",
                                    ui.input_select(
                                        id="tso_plot",
                                        label="",
                                        choices=tso_choices,
                                        selected='tso',
                                    ),
                                    class_="px-0 py-0 m-0",
                                ),
                            ),
                            'Resolution:',
                            ui.tooltip(
                                ui.input_numeric(
                                    id='tso_resolution',
                                    label='',
                                    value=0.0,
                                    min=0.0, max=6000.0, step=50.0,
                                ),
                                "True resolution = 6000",
                                id='tso_resolution_tooltip',
                                placement='bottom',
                            ),
                            width=1/2,
                            fixed_width=False,
                            gap='0px',
                            fill=False,
                            fillable=True,
                            class_="p-0 pb-1 m-0",
                        ),

                        ui.panel_conditional(
                            "input.tab === 'Noise'",
                            ui.layout_column_wrap(
                                "Wavelength:",
                                ui.input_select(
                                    "noise_wl_scale",
                                    label="",
                                    choices=wl_scales,
                                    selected='log',
                                ),
                                width=1/2,
                                fixed_width=False,
                                gap='0px',
                                fill=False,
                                fillable=True,
                                class_="p-0 m-0",
                            ),
                            class_="p-0 m-0",
                        ),
                        ui.panel_conditional(
                            "input.tab === 'TSO'",
                            "Wavelength:",
                            ui.layout_column_wrap(
                                ui.input_numeric(
                                    id='tso_wl_min', label='',
                                    value=None, min=0.2, max=30.0, step=0.02,
                                ),
                                ui.input_numeric(
                                    id='tso_wl_max', label='',
                                    value=None, min=0.2, max=30.0, step=0.1,
                                ),
                                ui.input_select(
                                    id="tso_wl_scale",
                                    label='',
                                    choices=wl_scales,
                                    selected='linear',
                                ),
                                width=1/3,
                                fixed_width=False,
                                gap='5px',
                                fill=False,
                                fillable=True,
                                class_="p-0 pb-1 m-0",
                            ),
                            "Depth:",
                            ui.layout_column_wrap(
                                ui.input_numeric(
                                    id='tso_depth_min',
                                    label='',
                                    value=None,
                                ),
                                ui.input_numeric(
                                    id='tso_depth_max',
                                    label='',
                                    value=None,
                                ),
                                ui.input_action_button(
                                    id="redraw_tso",
                                    label="Re-draw",
                                    class_="btn btn-outline-primary btn-sm",
                                ),
                                width=1/3,
                                fixed_width=False,
                                gap='5px',
                                fill=False,
                                fillable=True,
                                class_="p-0 pb-2 m-0",
                            ),
                            class_="px-0 py-0 m-0",
                        ),

                        class_="px-0 py-0 m-0",
                    ),

                    class_="px-2 py-1 pb-0 m-0 gap-0",
                    style=card_style,
                    fill=False,
                ),
                fill=False,
                class_="p-0 m-0 gap-0",
            ),
            # Search nearby Gaia targets for acquisition
            ui.card(
                ui.card_body(
                    ui.tooltip(
                        ui.markdown('**FOV targets**'),
                        'Gaia targets within 80" of science target',
                        id="gaia_tooltip",
                        placement="top",
                    ),
                    ui.layout_column_wrap(
                        ui.input_task_button(
                            id="search_gaia_ta",
                            label="Search nearby targets",
                            label_busy="processing...",
                            class_='btn btn-outline-secondary btn-sm',
                        ),
                        ui.input_action_button(
                            id="get_acquisition_target",
                            label="Print target data",
                            class_='btn btn-outline-secondary btn-sm',
                        ),
                        width=1,
                        heights_equal='row',
                        gap='7px',
                        class_="px-0 py-0 mx-0 my-0",
                    ),
                    class_="px-2 py-1 pb-2 m-0 gap-2",
                    style=card_style,
                    fill=False,
                ),
                fill=False,
            ),
            body_args=dict(class_="p-2 m-0"),
        ),

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Results
        ui.layout_columns(
            custom_card(
                ui.card_header("WALTzER runs"),
                # current setup and TSO runs
                ui.layout_columns(
                    # Left
                    ui.layout_column_wrap(
                        ui.input_select(
                            id="display_tso_run",
                            label='',
                            choices=make_tso_labels(tso_runs),
                            selected=[''],
                            width='100%',
                        ),
                        ui.input_task_button(
                            id="run_waltzer",
                            label="Run WALTzER",
                            label_busy="processing...",
                            width='100%',
                            style="border: 3px solid;",
                        ),
                        gap='10px',
                        width=1,
                        class_="px-0 py-0 mx-0 my-0",
                    ),
                    # TBD: Set disabled based on existing TSOs
                    ui.layout_column_wrap(
                        ui.input_action_button(
                            id="save_data_button",
                            label="Save data",
                            class_="btn btn-outline-success btn-sm",
                            width='110px',
                        ),
                        ui.input_action_button(
                            id="save_button",
                            label="Save TSO",
                            class_="btn btn-outline-success btn-sm",
                            disabled=False,
                            width='110px',
                        ),
                        ui.input_action_button(
                            id="delete_button",
                            label="Delete TSO",
                            class_='btn btn-outline-danger btn-sm',
                            disabled=False,
                            width='110px',
                        ),
                        width=1,
                        gap='5px',
                        class_="px-0 py-0 mx-0 my-0",
                    ),
                    col_widths=(9,3),
                    fill=True,
                    fillable=True,
                ),
                body_args=dict(class_="p-2 m-1"),
            ),
            # Display panels
            ui.navset_card_tab(
                ui.nav_panel(
                    "Stellar SED",
                    custom_card(
                        output_widget("plotly_sed", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='350px',
                    ),
                ),
                ui.nav_panel(
                    ui.output_text('transit_depth_label'),
                    custom_card(
                        output_widget("plotly_depth", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='350px',
                    ),
                    value='transit_tab',
                ),
                ui.nav_panel(
                    "Sky view",
                    custom_card(
                        ui.HTML(
                            '<iframe id="esasky" '
                            'height="100%" '
                            'width="100%" '
                            'style="overflow" '
                            'src="https://sky.esa.int/esasky/?target=0.0%200.0'
                            '&fov=0.2&sci=true&hide_welcome=true" '
                            'frameborder="0" allowfullscreen></iframe>',
                        ),
                        body_args=dict(class_='m-0 p-0'),
                        full_screen=True,
                        height='350px',
                    ),
                ),
                ui.nav_panel(
                    "Noise",
                    custom_card(
                        output_widget("plotly_variance", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='400px',
                    ),
                ),
                ui.nav_panel(
                    "TSO",
                    #tso_popover,
                    custom_card(
                        output_widget("plotly_tso", fillable=True),
                        body_args=dict(padding='0px'),
                        full_screen=True,
                        height='400px',
                    ),
                ),
                id="tab",
            ),
            # Text panel
            ui.navset_card_tab(
                ui.nav_panel(
                    "Results",
                    ui.span(
                        ui.output_ui(id="results"),
                        style="font-family: monospace; font-size:medium;",
                    ),
                ),
                ui.nav_panel(
                    "FOV targets",
                    ui.output_data_frame(id="acquisition_targets"),
                ),
            ),
            col_widths=[12, 12, 12],
            fill=False,
        ),
        col_widths=[3, 3, 6],
    ),
    title='WALTzER TSO',
    theme=f'{ROOT}/data/base_theme.css',
)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def server(input, output, session):
    actual_resolution = reactive.Value(6000)
    wl_binsize = reactive.Value(1)
    bookmarked_sed = reactive.Value(False)
    bookmarked_depth = reactive.Value(False)
    saturation_label = reactive.Value(None)
    update_catalog_flag = reactive.Value(False)
    update_sed_flag = reactive.Value(None)
    update_depth_flag = reactive.Value(None)
    uploaded_units = reactive.Value(None)
    acq_target_list = reactive.Value(None)
    current_acq_science_target = reactive.Value(None)
    preset_sed = reactive.Value(None)
    preset_obs_dur = reactive.Value(None)
    esasky_command = reactive.Value(None)
    programs_info = reactive.Value(None)
    clipboard = reactive.Value('')
    data_clipboard = reactive.Value(None)

    # Invisible flags
    @reactive.effect
    @reactive.event(bookmarked_sed)
    def _():
        value = len(bookmarked_spectra['sed']) > 0
        ui.update_switch('has_sed_bookmarks', value=value)


    @reactive.effect
    @reactive.event(bookmarked_depth)
    def _():
        obs_geometry = input.obs_geometry.get()
        value = len(bookmarked_spectra[obs_geometry]) > 0
        ui.update_switch('has_depth_bookmarks', value=value)


    @reactive.effect
    @reactive.event(input.main_settings)
    def _():
        with open(f'{ROOT}/data/last_updated_trexolist.txt', 'r') as f:
            last_trexo = f.readline().replace('_','-')
        with open(f'{ROOT}/data/last_updated_nea.txt', 'r') as f:
            last_nasa = f.readline().replace('_','-')

        waltz_ver = waltz.__version__
        github_url = "https://github.com/pcubillos/waltzer_etc"

        m = ui.modal(
            ui.hr(),
            ui.markdown(welcome),
            ui.span(
                ui.HTML('The Github repository is located here: '),
                ui.tags.a(
                    fa.icon_svg("github", fill='black'),
                    href=github_url,
                    target="_blank",
                ),
                ui.tags.a(github_url, href=github_url, target="_blank"),
            ),
            ui.hr(),
            # JWST and NASA databases:
            ui.markdown(f'JWST database last updated: {last_trexo}'),
            ui.markdown(f"NASA exoplanets database last updated: {last_nasa}"),
            ui.hr(),
            title=ui.markdown(f"**WALTzER ETC (version {waltz.__version__})**"),
            easy_close=True,
            size='l',
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.display_tso_run)
    def update_full_state():
        """
        When a user chooses a run from display_tso_run, update the entire
        front end to match the run setup.
        """
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]

        #detector = get_detector(inst, mode, detectors)

        ## The instrumental setting
        #choices = detector.get_constrained_val('filters')
        #ui.update_select(
        #    'filter',
        #    label=detector.filter_label,
        #    choices=choices,
        #    selected=filter,
        #)

        ## The target:
        #current_target = input.target.get()
        #current_tdur = _safe_num(input.t_dur.get(), default=2.0, cast=float)

        #name = tso['target']
        #t_dur = float(tso['transit_dur'])
        #planet_model_type = tso['planet_model_type']
        #ui.update_selectize('target', selected=name)
        #norm_band = tso['norm_band']
        #norm_mag = str(tso['norm_mag'])
        #sed_type = tso['sed_type']

        #if name != current_target:
        #    if name not in cache_target:
        #        cache_target[name] = {}
        #    cache_target[name]['t_eff'] = tso['t_eff']
        #    cache_target[name]['log_g'] = tso['log_g']
        #    cache_target[name]['t_dur'] = t_dur
        #    cache_target[name]['depth_label'] = tso['depth_label']
        #    cache_target[name]['rprs_sq'] = tso['rprs_sq']
        #    cache_target[name]['teq_planet'] = tso['teq_planet']
        #    cache_target[name]['norm_band'] = norm_band
        #    cache_target[name]['norm_mag'] = norm_mag
        #else:
        #    ui.update_numeric('t_eff', value=float(tso['t_eff']))
        #    ui.update_numeric('log_g', value=float(tso['log_g']))
        #    ui.update_numeric('t_dur', value=float(t_dur))
        #    ui.update_numeric('magnitude', value=float(norm_mag))

        ## sed_type, sed_model, norm_band, norm_mag, sed_label
        #ui.update_select('sed_type', selected=sed_type)
        #reset_sed = (
        #    sed_type != input.sed_type.get()
        #    or float(tso['t_eff']) != _safe_num(input.t_eff.get(), default=float(tso['t_eff']), cast=float)
        #    or float(tso['log_g']) != _safe_num(input.log_g.get(), default=float(tso['log_g']), cast=float)
        #)
        #if sed_type in sed_dict:
        #    if reset_sed:
        #        preset_sed.set(tso['sed_model'])
        #    else:
        #        choices = sed_dict[sed_type]
        #        selected = tso['sed_model']
        #        ui.update_select("sed", choices=choices, selected=selected)

        ## The observation
        #warning_text.set(tso['warnings'])
        #obs_geometry = tso['obs_geometry']
        #ui.update_select('obs_geometry', selected=obs_geometry)
        #if float(t_dur) != float(current_tdur):
        #    preset_obs_dur.set(tso['obs_dur'])
        #else:
        #    ui.update_numeric('obs_dur', value=float(tso['obs_dur']))

        #choices = depth_choices[obs_geometry]
        #ui.update_select(
        #    "planet_model_type", choices=choices, selected=planet_model_type,
        #)
        #if planet_model_type == 'Input':
        #    choices = list(spectra[obs_geometry])
        #    selected = tso['depth_label']
        #    ui.update_select("depth", choices=choices, selected=selected)
        #elif planet_model_type == 'Flat':
        #    ui.update_numeric("transit_depth", value=tso['rprs_sq'])
        #elif planet_model_type == 'Blackbody':
        #    ui.update_numeric("eclipse_depth", value=tso['rprs_sq'])
        #    ui.update_numeric("teq_planet", value=tso['teq_planet'])

        ## TSO plot popover menu
        #if tso['is_tso']:
        #    min_wl, max_wl = jwst._get_tso_wl_range(tso)
        #    ui.update_numeric('tso_wl_min', value=min_wl)
        #    ui.update_numeric('tso_wl_max', value=max_wl)

        #    resolution = int(_safe_num(input.tso_resolution.get(), default=250, cast=int))
        #    n_obs = int(_safe_num(input.n_obs.get(), default=1, cast=int))
        #    units = 'percent'  if obs_geometry=='transit' else 'ppm'
        #    ui.update_select('tso_depth_units', selected=units)
        #    min_depth, max_depth, step = jwst._get_tso_depth_range(
        #        tso, resolution, units,
        #    )
        #    ui.update_numeric('tso_depth_min', value=min_depth, step=step)
        #    ui.update_numeric('tso_depth_max', value=max_depth, step=step)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Instrument
    @reactive.effect
    @reactive.event(input.run_waltzer)
    def _():
        name = input.target.get()

        # Get the variances first
        efficiency = input.efficiency.get() * pc.percent
        bands = list(input.bands.get())
        n_obs = input.n_obs.get()

        obs_geometry = input.obs_geometry.get()
        transit_dur = input.t_dur.get()
        obs_dur = input.obs_dur.get()

        if obs_dur < transit_dur:
            error_msg = ui.markdown(
                f"**Warning:**<br>observation duration is shorter than "
                f"the {obs_geometry} duration"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            return

        if len(bands) == 0:
            # Nothing to calculate
            success = ui.markdown("Please select **at least one** band!")
            ui.notification_show(success, type="warning", duration=5)
            return

        # SED flux
        sed_type, sed_model, norm_mag, sed_label = parse_sed(input, spectra)
        if sed_label in bookmarked_spectra['sed']:
            flux = spectra['sed'][sed_label]['flux']
        else:
            sed_flux = load_sed(sed_model, sed_type, cache_seds)
            flux = sed.normalize_vega(wl, sed_flux, norm_mag)
            spectra['sed'][sed_label] = {
                'wl': wl, 'flux': flux, 'filename':None,
            }
            bookmarked_spectra['sed'].append(sed_label)
            bookmarked_sed.set(True)

        # Target setup:
        planet_model_type, depth_label, rprs_sq, teq_planet = parse_obs(input)

        # Electrons per second from each source
        tso = {}
        for band in bands:
            det = detectors[band]
            variances = det.calc_noise(wl, flux)
            total_variance = np.sum(variances, axis=0)
            band_flux = variances[0]
            # TBD this should mirror waltzer_sample()'s tso dictionary
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

        tso_label = make_tso_label(input, spectra)

        tso['meta'] = {
            'bands': bands,
            'efficiency': efficiency,
            'n_obs': n_obs,
            'transit_dur': transit_dur,
            # GUI-only meta
            'target': name,
            'sed_model': sed_model,
            'obs_dur': obs_dur,
            'obs_geometry': obs_geometry,
            'label': tso_label,
            'planet_model_type': planet_model_type,
            'depth_label': depth_label,
            'rprs_sq': rprs_sq,
            'teq_planet': teq_planet,
        }

        run_type = obs_geometry.capitalize()
        tso_runs[run_type][tso_label] = tso
        tso_labels = make_tso_labels(tso_runs)
        ui.update_select(
            'display_tso_run',
            choices=tso_labels,
            selected=f'{run_type}_{tso_label}',
        )

        success = "WALTzER TSO model simulated!"
        ui.notification_show(success, type="message", duration=2)
        print('~~ WALTzER TSO done! ~~')
        return


    @reactive.effect
    @reactive.event(input.delete_button)
    def _():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        del tso_runs[key][tso_label]
        tso_labels = make_tso_labels(tso_runs)
        ui.update_select('display_tso_run', choices=tso_labels)


    @reactive.effect
    @reactive.event(input.save_data_button)
    def _():
        # Make a filename from current TSO
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]
        target = tso['meta']['target'].replace(' ', '')
        filename = f'{key.lower()}_{target}_waltzer_tso.dat'
        savefile = Path(f'{current_dir}/{filename}')

        labs, data = data_clipboard.get()
        np.savetxt(
            savefile,
            data,
            header=labs,
        )

        ui.notification_show(
            f"TSO model saved to file: '{savefile}'",
            type="message",
            duration=5,
        )

    @reactive.effect
    @reactive.event(input.save_button)
    def _():
        # Make a filename from current TSO
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        filename = f'{key.lower()}_waltzer_tso.pickle'

        m = ui.modal(
            ui.input_text(
                id='tso_save_file',
                label='Save TSO run to this file:',
                value=filename,
                width='100%',
            ),
            ui.input_text(
                id='tso_save_dir',
                label='Located in this folder:',
                value=current_dir,
                placeholder='select a folder',
                width='100%',
            ),
            ui.input_action_button(
                id='tso_save_button',
                label='Save to file',
            ),
            title="Download TSO run",
            easy_close=True,
            size='l',
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.tso_save_button)
    def _():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]

        # Update meta
        efficiency = input.efficiency.get() * pc.percent
        n_obs = input.n_obs.get()
        transit_dur = input.t_dur.get()
        tso['meta']['efficiency'] = efficiency
        tso['meta']['n_obs'] = n_obs
        tso['meta']['transit_dur'] = transit_dur

        folder = input.tso_save_dir.get().strip()
        if folder == '':
            folder = '.'

        filename = input.tso_save_file.get()
        if filename.strip() == '':
            filename = f'tso_{key.lower()}_run.pickle'

        savefile = Path(f'{folder}/{filename}')
        if savefile.suffix == '':
            savefile = savefile.parent / f'{savefile.name}.pickle'

        tso_run = {
            tso['meta']['target']: tso,
        }
        with open(savefile, 'wb') as handle:
            pickle.dump(tso_run, handle, protocol=4)

        ui.modal_remove()
        ui.notification_show(
            f"TSO model saved to file: '{savefile}'",
            type="message",
            duration=5,
        )


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Target
    @reactive.Effect
    @reactive.event(input.target_filter, update_catalog_flag)
    def _():
        update_catalog_flag.get()
        mask = np.zeros(nplanets, bool)
        if 'jwst' in input.target_filter.get():
            mask |= is_jwst
        if 'transit' in input.target_filter.get():
            mask |= is_transit
        if 'non_transit' in input.target_filter.get():
            mask |= ~is_transit
        if 'tess' in input.target_filter.get():
            mask |= ~is_confirmed

        targets = [
            target.planet for target,flag in zip(catalog.targets,mask)
            if flag
        ]
        for i,target in enumerate(catalog.targets):
            if mask[i]:
                targets += target.aliases

        # Preserve current target if possible:
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        planet = '' if target is None else target.planet
        ui.update_selectize('target', choices=targets, selected=planet)


    @reactive.effect
    @reactive.event(input.show_info)
    def _():
        """
        Display system parameters
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        planet_info, star_info, aliases = pretty_print_target(target)
        clipboard.set(target.machine_readable_text())

        info = ui.layout_columns(
            ui.span(planet_info, style="font-family: monospace;"),
            ui.span(star_info, style="font-family: monospace;"),
            width=1/2,
        )

        m = ui.modal(
            info,
            ui.span(aliases, style="font-family: monospace;"),
            title=ui.markdown(f'System parameters for: **{target.planet}**'),
            size='l',
            easy_close=True,
            footer=ui.input_action_button(
                id="copy_planet",
                label="Copy to clipboard",
                class_='btn btn-sm',
            ),
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.copy_planet)
    async def copy_clipboard():
        await session.send_custom_message(
            "copy_to_clipboard",
            clipboard.get(),
        )

    @reactive.effect
    @reactive.event(input.copy_status)
    def notify_copy_status():
        status = input.copy_status()
        msg_type = 'message' if status == "success" else 'warning'
        if status == "success":
            msg = "Copied to clipboard!"
        elif status == "failure":
            msg = "Failed to copy!"
        ui.notification_show(msg, type=msg_type, duration=4)


    @reactive.effect
    @reactive.event(input.show_observations)
    def _():
        """
        Display JWST observations
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None or not hasattr(target, 'programs'):
            return

        programs_info.set(target.programs)

        keys = ui.HTML(
            'Keys:<br>'
            '<span style="color:#0B980D">Observed, publicly available.</span>'
            '<br><span style="color:#FFa500">Observed, in proprietary period.</span><br>'
            '<span>To be observed, planned window.</span>'
            '<br><span style="color:red">Failed, withdrawn, or skipped.</span>'
        )

        m = ui.modal(
            ui.output_data_frame('trexo_df'),
            ui.hr(),
            keys,
            title=ui.markdown(f'JWST programs for: **{target.host}**'),
            size='xl',
            easy_close=True,
        )
        ui.modal_show(m)


    @render.data_frame
    @reactive.event(programs_info)
    def trexo_df():
        data = programs_info.get()
        nobs = len(data)

        today = datetime.today()
        date_obs = [obs['date_start'] for obs in data]
        plan_obs = [obs['plan_window'] for obs in data]
        propriety = [obs['proprietary_period'] for obs in data]
        warnings = [
            i for i,obs in enumerate(data)
            if obs['status'] in ['Skipped', 'Failed', 'Withdrawn']
        ]
        available = []
        private = []
        dates = []
        tbd_dates = []
        for i in range(nobs):
            if isinstance(date_obs[i], datetime):
                release = date_obs[i] + timedelta(days=365.0*propriety[i]/12)
                if release < today:
                    available.append(i)
                else:
                    private.append(i)
                dates.append(
                    date_obs[i].strftime('%Y-%m-%d') +
                    f' ({propriety[i]} m)'
                )
            else:
                if plan_obs[i] is None:
                    dates.append(f'--- ({propriety[i]} m)')
                elif '-' in plan_obs[i]:
                    date = plan_obs[i][0:plan_obs[i].index(' ')]
                    dates.append(f'{date} ({propriety[i]} m)')
                else:
                    dates.append(f'{plan_obs[i]} ({propriety[i]} m)')

                if i not in warnings:
                    tbd_dates.append(i)
        styles = [
            {
                'rows': available,
                'style': {"color": "#0B980D"},
            },
            {
                'rows': warnings,
                'style': {"color": "red"},
            },
            {
                'rows': private,
                'style': {"color": "#FFa500"},
            },
        ]
        programs = [
            ui.tags.a(
                f"{obs['category']} {obs['pid']}",
                href=stsci_url.replace('PID', obs['pid']),
                target="_blank",
            )
            for obs in data
        ]
        planets = [', '.join(obs['planets']) for obs in data]

        data_df = {
            'Program ID': programs,
            'PI': [obs['pi'] for obs in data],
            'Target name': [obs['target'] for obs in data],
            'Planet(s)': planets,
            'Event': [obs['event'] for obs in data],
            'Status': [obs['status'] for obs in data],
            'Instrument': [obs['instrument'] for obs in data],
            'Mode': [obs['mode'] for obs in data],
            'Disperser': [obs['disperser'] for obs in data],
            'Filter': [obs['filter'] for obs in data],
            'Subarray': [obs['subarray'] for obs in data],
            'Readout': [obs['readout'] for obs in data],
            'Groups': [obs['groups'] for obs in data],
            'Duration (h)': [obs['duration'] for obs in data],
            'Obs date (prop. period)': dates,
        }
        df = pd.DataFrame(data=data_df)

        return render.DataGrid(
            df,
            styles=styles,
            width='100%',
        )


    @reactive.effect
    @reactive.event(input.target)
    def update_target_icons():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)

        # JWST flag
        if target is None:
            icon_color = 'gray'
            tip = 'not a JWST target (yet)'
        elif target.is_jwst_planet:
            icon_color = '#FAC205'
            tip = 'This is a JWST target'
        elif target.is_jwst_host:
            icon_color = 'goldenrod'
            tip = ui.markdown("This *host* is a JWST target")
        else:
            icon_color = 'gray'
            tip = 'not a JWST target (yet)'
        icon = fa.icon_svg("circle-info", fill=icon_color)
        ui.update_action_link('show_observations', icon=icon)
        ui.update_tooltip('jwst_tooltip', tip)

        if target is None:
            return

        # Confirmed or candidate
        ui.update_switch('is_candidate', value=not target.is_confirmed)

        # Aliases
        if len(target.aliases) > 0:
            aliases = ', '.join(target.aliases)
            info_label = f"Also known as: {aliases}"
        else:
            info_label = 'System info'
        ui.update_tooltip('target_info_tooltip', info_label)


    @reactive.effect
    @reactive.event(input.target)
    async def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return ''

        url = f'{nasa_url}/{target.planet}'
        await session.send_custom_message(
            "set_nasa_href",
            {"url": url}
        )


    @reactive.effect
    @reactive.event(input.target)
    def _():
        """Update target properties"""
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return
        if name in target.aliases:
            ui.update_selectize('target', selected=target.planet)

        # Physical properties:
        if target.planet in cache_target:
            t_eff  = cache_target[target.planet]['t_eff']
            log_g = cache_target[target.planet]['log_g']
            t_dur = cache_target[target.planet]['t_dur']
            magnitude = cache_target[target.planet]['norm_mag']
        else:
            t_eff = as_str(target.teff, '.1f', '')
            log_g = as_str(target.logg_star, '.2f', '')
            t_dur = as_str(target.transit_dur, '.3f', '')
            magnitude = f'{target.v_mag:.3f}'

        ui.update_numeric('t_eff', value=float(t_eff))
        ui.update_numeric('log_g', value=float(log_g))
        ui.update_numeric('magnitude', value=float(magnitude))
        if t_dur == '':
            t_dur = '0.0'
        ui.update_numeric('t_dur', value=float(t_dur))

        delete_catalog = {
            "event": 'deleteCatalogue',
            "content": { 'overlayName': 'Nearby Gaia sources'}
        }
        delete_footprint = {
            "event": 'deleteFootprintsOverlay',
            "content": {'overlayName': 'visit splitting distance'}
        }
        goto = {
            "event": "goToRaDec",
            "content":{"ra": f"{target.ra}", "dec": f"{target.dec}"}
        }
        esasky_command.set([delete_catalog, delete_footprint, goto])

        # Observing properties:
        if name in cache_target and cache_target[name]['rprs_sq'] is not None:
            rprs_square_percent = cache_target[name]['rprs_sq']
            teq_planet = cache_target[name]['teq_planet']
            cache_target[name]['rprs_sq'] = None
            cache_target[name]['teq_planet'] = None
        else:
            teq_planet = np.round(target.eq_temp, decimals=1)
            if np.isnan(teq_planet):
                teq_planet = 0.0
            rprs_square = target.rprs**2.0
            if np.isnan(rprs_square):
                rprs_square = 0.0
            rprs_square_percent = np.round(100*rprs_square, decimals=4)

        if rprs_square_percent is not None:
            ui.update_numeric("transit_depth", value=rprs_square_percent)
            ui.update_numeric("eclipse_depth", value=rprs_square_percent)
        if teq_planet is not None:
            ui.update_numeric('teq_planet', value=teq_planet)


    @reactive.Effect
    @reactive.event(input.sed_type, input.t_eff, input.log_g, update_sed_flag)
    def choose_sed():
        sed_type = input.sed_type.get()
        if sed_type in sed_dict:
            choices, selected = get_auto_sed(input)
            if preset_sed.get() is not None:
                selected = preset_sed.get()
                preset_sed.set(None)
        elif sed_type == 'blackbody':
            t_eff = _safe_num(input.t_eff.get(), default=0.0, cast=float)
            selected = f' Blackbody (Teff={t_eff:.0f} K)'
            choices = [selected]
        elif sed_type == 'input':
            # bookmarking SED create spectra['sed'] items,
            # a None filename flags them out of the 'input' list
            choices = [
                sed for sed,model in spectra['sed'].items()
                if model['filename'] is not None
            ]
            selected = None

        ui.update_select("sed", choices=choices, selected=selected)


    @reactive.effect
    @reactive.event(
        bookmarked_sed, input.sed, input.magnitude,
        #input.t_eff, (blackbody SED)
    )
    def update_sed_book_icon():
        """Check current SED is bookmarked"""
        sed_type, sed_model, norm_mag, sed_label = parse_sed(input, spectra)
        is_bookmarked = sed_label in bookmarked_spectra['sed']
        bookmarked_sed.set(is_bookmarked)
        if is_bookmarked:
            icon = fa.icon_svg("star", style='solid', fill='gold')
        else:
            icon = fa.icon_svg("star", style='regular', fill='black')
        ui.update_action_link('bookmark_sed', icon=icon)


    @reactive.Effect
    @reactive.event(input.bookmark_sed)
    def _():
        """Toggle bookmarked SED"""
        sed_type, sed_model, norm_mag, sed_label = parse_sed(input, spectra)
        if sed_type is None:
            msg = ui.markdown("**Error**:<br>No SED model to bookmark")
            ui.notification_show(msg, type="error", duration=5)
            return
        is_bookmarked = not bookmarked_sed.get()
        bookmarked_sed.set(is_bookmarked)
        if is_bookmarked:
            sed_flux = load_sed(sed_model, sed_type, cache_seds)
            # Normalize according to Vmag
            flux = sed.normalize_vega(wl, sed_flux, norm_mag)

            spectra['sed'][sed_label] = {
                'wl': wl, 'flux': flux, 'filename':None,
            }
            bookmarked_spectra['sed'].append(sed_label)
        else:
            bookmarked_spectra['sed'].remove(sed_label)

    @reactive.Effect
    @reactive.event(input.clear_sed_bookmarks)
    def _():
        """Clear all bookmarked SEDs"""
        bookmarked_spectra['sed'].clear()
        bookmarked_sed.set(False)
        update_sed_flag.set('cleared')  # trigger UI updates
        ui.notification_show("Cleared all SED bookmarks", type="message", duration=3)


    @render.ui
    @reactive.event(input.obs_geometry)
    def depth_label():
        """Set depth model label"""
        obs_geometry = input.obs_geometry.get()
        return ui.HTML(f'<b>{obs_geometry.capitalize()} spectra</b> '),


    @reactive.effect
    @reactive.event(input.obs_geometry)
    def _():
        """Set depth model label"""
        geom = input.obs_geometry.get()
        ui.update_tooltip('depth_up_tooltip', f'Upload {geom} depth model')
        ui.update_tooltip('depth_book_tooltip', f'Bookmark {geom} depth model')
        ui.update_tooltip('depth_clear_tooltip', f'Clear all {geom} depth model')


    @reactive.effect
    @reactive.event(
        bookmarked_depth, input.obs_geometry, input.planet_model_type,
        input.depth, input.transit_depth, input.eclipse_depth, input.teq_planet,
    )
    def update_depth_book_icon():
        """Check current transit-depth is bookmarked"""
        obs_geometry = input.obs_geometry.get()
        depth_label = planet_model_name(input)
        is_bookmarked = depth_label in bookmarked_spectra[obs_geometry]
        bookmarked_depth.set(is_bookmarked)

        fill = 'royalblue' if is_bookmarked else 'gray'
        icon = fa.icon_svg("earth-americas", style='solid', fill=fill)
        ui.update_action_link('bookmark_depth', icon=icon)


    @reactive.Effect
    @reactive.event(input.bookmark_depth)
    def _():
        """Toggle bookmarked depth model"""
        obs_geometry = input.obs_geometry.get()
        depth_label = planet_model_name(input)
        planet_model_type = input.planet_model_type.get()
        if depth_label is None:
            msg = ui.markdown(
                f"**Error:**<br>No {obs_geometry} depth model to bookmark"
            )
            ui.notification_show(msg, type="error", duration=5)
            return
        is_bookmarked = not bookmarked_depth.get()
        bookmarked_depth.set(is_bookmarked)
        if is_bookmarked:
            bookmarked_spectra[obs_geometry].append(depth_label)
            depth_label, wl, depth = read_depth_spectrum(input, spectra)
            if planet_model_type != 'Input':
                spectra[obs_geometry][depth_label] = {'wl': wl, 'depth': depth}
        else:
            bookmarked_spectra[obs_geometry].remove(depth_label)

    @reactive.Effect
    @reactive.event(input.clear_depth_bookmarks)
    def _():
        """Clear bookmarked depth models for the current geometry"""
        obs_geometry = input.obs_geometry.get()
        bookmarked_spectra[obs_geometry].clear()
        bookmarked_depth.set(False)
        update_depth_flag.set('cleared')  # trigger UI updates
        ui.notification_show(f"Cleared all {obs_geometry} depth bookmarks", type="message", duration=3)


    @reactive.effect
    @reactive.event(input.obs_geometry, update_depth_flag)
    def update_depth_types_and_models():
        obs_geometry = input.obs_geometry.get()
        choices = depth_choices[obs_geometry]
        model_type = input.planet_model_type.get()
        if model_type not in choices:
            model_type = choices[0]
        ui.update_select(
            "planet_model_type", choices=choices, selected=model_type,
        )

        models = list(spectra[obs_geometry])
        name = input.target.get()
        cached = (
            name in cache_target and
            cache_target[name]['depth_label'] is not None
        )
        selected = input.depth.get()
        if cached:
            selected = cache_target[name]['depth_label']
            cache_target[name]['depth_label'] = None
        elif selected not in models:
            selected = None if len(models) == 0 else models[0]
        ui.update_select("depth", choices=models, selected=selected)

        if len(models) > 0:
            tooltip_text = ''
        elif obs_geometry == 'transit':
            tooltip_text = f'Upload a {obs_geometry} depth spectrum'
        elif obs_geometry == 'eclipse':
            tooltip_text = f'Upload an {obs_geometry} depth spectrum'
        ui.update_tooltip('depth_tooltip', tooltip_text)


    @render.text
    @reactive.event(input.obs_geometry)
    def transit_dur_label():
        obs_geometry = input.obs_geometry.get().capitalize()
        return f"{obs_geometry[0]}_dur (h):"

    @render.text
    @reactive.event(input.obs_geometry)
    def transit_depth_label():
        obs_geometry = input.obs_geometry.get().capitalize()
        return f"{obs_geometry} depth"

    #@render.ui
    #@reactive.event(warning_text)
    #def warnings_label():
    #    warnings = warning_text.get()
    #    if len(warnings) == 0:
    #        return "Warnings"
    #    n_warn = len(warnings)
    #    return ui.HTML(f'<div style="color:red;">Warnings ({n_warn})</div>')

    @reactive.Effect
    @reactive.event(
        input.t_dur, input.settling_time, input.baseline_time,
        input.min_baseline_time,
    )
    def _():
        """Set observation time based on transit dur and popover settings"""
        if preset_obs_dur.get() is not None:
            obs_dur = preset_obs_dur.get()
            preset_obs_dur.set(None)
            ui.update_numeric('obs_dur', value=float(f'{obs_dur:.2f}'))
            return
        t_dur_val = _safe_num(req(input.t_dur).get(), default=0.0, cast=float)
        if t_dur_val == 0.0:
            ui.update_numeric('obs_dur', value=0.0)
            return
        transit_dur = t_dur_val
        settling = req(input.settling_time).get()

        baseline = req(input.baseline_time).get()
        min_baseline = req(input.min_baseline_time).get()
        baseline = np.clip(baseline*transit_dur, min_baseline, np.inf)
        # Tdwell = T_settle + T14 + 2*max(1, T14/2)
        obs_dur = settling + transit_dur + 2.0*baseline
        ui.update_numeric('obs_dur', value=float(f'{obs_dur:.2f}'))


    @reactive.effect
    @reactive.event(input.upload_sed)
    def _():
        m = ui.modal(
            ui.markdown(
                "Input files must be plan-text files with two columns, "
                "the first one being the wavelength (microns) and "
                "the second one the stellar SED.<br>**Make sure "
                "the input units are correct before uploading a file!**"
            ),
            ui.input_radio_buttons(
                id="upload_wl_units",
                label='Wavelength units:',
                choices=wl_units,
                selected='micron',
                width='100%',
            ),
            ui.input_radio_buttons(
                id="upload_units",
                label='Flux units:',
                choices=sed_units,
                width='100%',
            ),
            ui.input_file(
                id="upload_file",
                label='',
                button_label="Browse",
                multiple=True,
                width='100%',
            ),
            title="Upload stellar spectrum",
            easy_close=True,
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.upload_depth)
    def _():
        obs_geometry = input.obs_geometry.get()
        m = ui.modal(
            ui.markdown(
                "Input files must be plan-text files with two columns, "
                "the first one being the wavelength (microns) and "
                f"the second one the {obs_geometry} depth.<br>**Make sure "
                "the input units are correct before uploading a file!**"
            ),
            ui.input_radio_buttons(
                id="upload_wl_units",
                label='Wavelength units:',
                choices=wl_units,
                selected='micron',
                width='100%',
            ),
            ui.input_radio_buttons(
                id="upload_units",
                label='Depth units:',
                choices=depth_units,
                width='100%',
            ),
            ui.input_file(
                id="upload_file",
                label='',
                button_label="Browse",
                multiple=True,
                width='100%',
            ),
            title="Upload planetary spectrum",
            easy_close=True,
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.upload_units)
    def _():
        uploaded_units.set(input.upload_units.get())


    @reactive.effect
    @reactive.event(input.upload_file)
    def _():
        new_model = input.upload_file.get()
        if not new_model:
            return

        # The units tell this function SED or depth spectrum:
        filename = new_model[0]['name']
        filepath = new_model[0]['datapath']

        wl_units = input.upload_wl_units.get()
        units = uploaded_units.get()

        _, wl, model = read_spectrum_file(
            filepath, units, wl_units, on_fail='warning',
        )
        if wl is None:
            msg = ui.markdown(
                f'**Error:**<br>Invalid format for input file:<br>*{filename}*'
            )
            ui.notification_show(msg, type="error", duration=7)
            return
        label = os.path.splitext(filename)[0]

        if units in depth_units:
            obs_geometry = input.obs_geometry.get()
            spectra[obs_geometry][label] = {
                'wl': wl,
                'depth': model,
                'units': units,
                'filename': f'input_{filename}',
            }
            if label not in bookmarked_spectra[obs_geometry]:
                bookmarked_spectra[obs_geometry].append(label)
            if input.planet_model_type.get() != 'Input':
                return
            # Trigger update choose_depth
            update_depth_flag.set(label)
        elif units in sed_units:
            spectra['sed'][label] = {
                'wl': wl,
                'flux': model,
                'units': units,
                'filename': f'input_{filename}',
            }
            update_sed_flag.set(label)


    @reactive.effect
    @reactive.event(input.tso_resolution)
    def update_actual_resolution():
        resolution = input.tso_resolution.get()
        if resolution == 0.0:
            binsize = 1
            actual_resolution.set(6000.0)
        else:
            idx = searchsorted_closest(resolutions, resolution)
            binsize = bins[idx]
            actual_resolution.set(resolutions[idx])

        if binsize != wl_binsize.get():
            wl_binsize.set(binsize)
        tip = f'True resolution = {actual_resolution.get():.1f}'
        ui.update_tooltip('tso_resolution_tooltip', tip)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Viewers
    @render_plotly
    def plotly_sed():
        bookmarked_sed.get() # (make panel reactive to remove all bookmarks)
        update_sed_flag.get()
        input.bookmark_sed.get()  # (make panel reactive to bookmark_sed)
        bands = list(input.bands.get())
        throughput = masked_throughput(bands)

        # Gather bookmarked SEDs
        model_names = bookmarked_spectra['sed']
        if len(model_names) == 0:
            fig = go.Figure()
            fig.update_layout(title='Bookmark SEDs models to show them here')
            return fig

        sed_models = []
        for model_name in model_names:
            model = dict(spectra['sed'][model_name])
            plot_model = dict(
                flux=model['flux'],
                wl=model['wl'],
            )
            sed_models.append(plot_model)
        current_model = parse_sed(input, spectra)[-1]

        wl_scale = input.plot_sed_xscale.get()
        flux_scale = input.plot_sed_yscale.get()
        wl_range = [input.sed_wl_min.get(), input.sed_wl_max.get()]
        units = 'mJy'

        resolution = input.plot_sed_resolution.get()

        fig = plt.plotly_sed_spectra(
            sed_models, model_names, current_model,
            units=units,
            wl_range=wl_range, wl_scale=wl_scale,
            flux_scale=flux_scale,
            resolution=resolution,
            throughput=throughput,
        )
        return fig


    @render_plotly
    def plotly_depth():
        input.bookmark_depth.get()  # (make panel reactive to bookmark_depth)
        bands = list(input.bands.get())
        throughput = masked_throughput(bands)

        update_depth_flag.get()
        obs_geometry = input.obs_geometry.get()
        model_names = bookmarked_spectra[obs_geometry]
        nmodels = len(model_names)
        if nmodels == 0:
            fig = go.Figure()
            title = f'Bookmark {obs_geometry.lower()} models to show them here'
            fig.update_layout(title=title)
            return fig

        current_model = planet_model_name(input)
        wl_scale = input.plot_depth_xscale.get()
        wl_range = [input.depth_wl_min.get(), input.depth_wl_max.get()]
        units = input.plot_depth_units.get()
        resolution = input.depth_resolution.get()

        depth_models = [spectra[obs_geometry][model] for model in model_names]
        fig = plt.plotly_depth_spectra(
            depth_models, model_names, current_model,
            units=units,
            wl_range=wl_range, wl_scale=wl_scale,
            resolution=resolution,
            obs_geometry=obs_geometry,
            throughput=throughput,
        )
        return fig


    @render_plotly
    def plotly_variance():
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return go.Figure()

        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]
        bands = tso['meta']['bands']

        plot_type = input.noise_plot.get()
        binsize = wl_binsize.get()
        wl_scale = input.noise_wl_scale.get()

        if plot_type == 'variance':
            head = 'wl(um)  source(e/s)  sky(e/s)  dark(e/s)  read_noise(e/s)  wl_half_width(um)'
            wl = np.hstack([tso[band]['wl'] for band in bands])
            var = np.hstack([tso[band]['variances'] for band in bands])
            hw = np.hstack([tso[band]['half_widths'] for band in bands])
            clip = np.vstack((wl, var, hw)).T
            data_clipboard.set([head, clip])

            fig = plt.plotly_variances(
                tso,
                wl_scale=wl_scale,
                binsize=binsize,
            )
            return fig

        # elif plot_type == 'snr':
        efficiency = input.efficiency.get() * pc.percent
        n_obs = input.n_obs.get()
        transit_dur = input.t_dur.get()
        obs_dur = input.obs_dur.get()
        obs_geometry = input.obs_geometry.get()

        # Read model
        depth_label, wl, depth = read_depth_spectrum(input, spectra)
        depth_model = wl, depth

        flux_data = waltz.simulate_fluxes(
            tso, depth_model, obs_geometry,
            n_obs, transit_dur, obs_dur, binsize,
            efficiency=efficiency,
        )

        head = 'wl(um)  flux_in(e)  flux_out(e)  var_in(e)  var_out(e)  wl_half_width(um) depth'
        clip = np.vstack([
            np.hstack([d for d in f_data])
            for f_data in flux_data[1:]
        ]).T
        data_clipboard.set([head, clip])

        fig = plt.plotly_flux_snr(
            flux_data,
            wl_scale=wl_scale,
            y_scale='log',
            y_label='time-integrated flux SNR',
        )
        return fig


    @render_plotly
    def plotly_tso():
        input.redraw_tso.get()
        tso_key = input.display_tso_run.get()
        if tso_key is None:
            return go.Figure()
        key, tso_label = tso_key.split('_', maxsplit=1)
        tso = tso_runs[key][tso_label]

        plot_type = input.tso_plot.get()
        binsize = wl_binsize.get()
        wl_scale = input.tso_wl_scale.get()

        efficiency = input.efficiency.get() * pc.percent
        n_obs = input.n_obs.get()

        obs_geometry = input.obs_geometry.get()
        transit_dur = input.t_dur.get()
        obs_dur = input.obs_dur.get()

        if obs_dur < transit_dur:
            error_msg = ui.markdown(
                f"**Warning:**<br>observation duration is shorter than "
                f"the {obs_geometry} duration"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            return go.Figure()

        # Transit-depth model at WALTzER resolving power
        depth_label, wl, depth = read_depth_spectrum(input, spectra)
        depth_model = waltz_model(wl, depth)

        tso_data = waltz.simulate_spectrum(
            tso, depth_model, obs_geometry,
            n_obs, transit_dur, obs_dur, binsize,
            efficiency=efficiency,
        )

        head = 'wl(um)  depth  depth_error  wl_half_width(um)'
        clip = np.vstack([
            np.hstack([d for d in t_data])
            for t_data in tso_data[1:]
        ]).T
        data_clipboard.set([head, clip])

        wl_scale = input.tso_wl_scale.get()
        wl_min = input.tso_wl_min.get()
        wl_max = input.tso_wl_max.get()
        wl_range = [wl_min, wl_max]

        if plot_type == 'tso':
            depth_units = 'percent'
            depth_range = [input.tso_depth_min.get(), input.tso_depth_max.get()]
            fig = plt.plotly_tso_spectra(
                tso, tso_data, depth_model,
                model_label='model',
                wl_range=wl_range, wl_scale=wl_scale,
                depth_units=depth_units, depth_range=depth_range,
                obs_geometry=obs_geometry,
            )

        elif plot_type == 'snr':
            fig = plt.plotly_depth_snr(
                tso,
                tso_data,
                wl_range=wl_range,
                wl_scale=wl_scale,
                y_scale='log',
                y_label = f'{obs_geometry}-depth SNR'
            )

        #elif plot_type == 'uncertainties':
        #    fig = plots.plotly_tso_fluxes(
        #        tso,
        #        wl_range=wl_range, wl_scale=wl_scale,
        #        obs_geometry=obs_geometry,
        #    )
        return fig


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Results
    @render.ui
    def results():
        # Only read for reactivity reasons:
        saturation_label.get()

        #config = parse_instrument(input)
        #config = None
        #if config is None:
        #    return ui.HTML('<pre> </pre>')

        name = input.target.get()
        #target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        #cached_target = (
        #    target is not None and
        #    target.host in cache_acquisition and
        #    cache_acquisition[target.host]['selected'] is not None
        #)

        #depth_label = parse_obs(input)[1]
        #transit_dur = _safe_num(input.t_dur.get(), default=2.0, cast=float)

        #if parse_sed(input, spectra)[-1] is None:
        #    return ui.HTML('<pre> </pre>')

        #obs_geometry = input.obs_geometry.get()
        #run_type = obs_geometry.capitalize()

        #report_text = jwst._print_pandeia_exposure()
        #target_name = f': {target.planet}' if target is not None else ''
        #report_text = f'<b>target{target_name}</b><br>{report_text}'

        #sed_type, sed_model, norm_mag, sed_label = parse_sed(
        #    input, spectra,
        #)
        #pixel_rate, full_well = get_saturation_values(
        #    filter, sed_label, norm_mag, cache_saturation,
        #)
        #if pixel_rate is not None:
        #    saturation_text = jwst._print_pandeia_saturation(
        #        pixel_rate, full_well,
        #        format='html',
        #    )
        #    report_text += f'<br>{saturation_text}'

        #tso_label = make_obs_label()
        #if tso_label in tso_runs[run_type]:
        #    tso_run = tso_runs[run_type][tso_label]
        #    if transit_dur == tso_run['transit_dur']:
        #        report_text += f'<br><br>{tso_run["stats"]}'
        report_text = name
        return ui.HTML(f'<pre>{report_text}</pre>')



    @reactive.effect
    @reactive.event(esasky_command)
    async def _():
        commands = esasky_command.get()
        if commands is None:
            return
        if not isinstance(commands, list):
            commands = [commands]
        for command in commands:
            command = json.dumps(command)
            await session.send_custom_message(
                "update_esasky",
                {"command": command},
            )


    @reactive.Effect
    @reactive.event(input.search_gaia_ta)
    def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return

        query = fetch_gaia_targets(
            target.ra, target.dec, max_separation=80.0, raise_errors=False,
        )
        if isinstance(query, str):
            msg = ui.HTML(
                "The Gaia astroquery request failed :(<br>"
                "Check the terminal for more info"
            )
            ui.notification_show(msg, type="error", duration=5)
            return

        cache_acquisition[target.host] = {'targets': query, 'selected': None}
        acq_target_list.set(query)
        current_acq_science_target.set(name)
        success = "Nearby targets found!  Open the '*FOV targets*' tab"
        ui.notification_show(ui.markdown(success), type="message", duration=5)

        circle = esasky_js_circle(target.ra, target.dec, radius=80.0)
        ta_catalog = esasky_js_catalog(query)
        esasky_command.set([ta_catalog, circle])


    @render.data_frame
    @reactive.event(acq_target_list, input.target)
    def acquisition_targets():
        """
        Display TA list, gets triggered only when tab is shown
        """
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return render.DataGrid(pd.DataFrame([]))
        if target.host not in cache_acquisition:
            # Do I need to? gets wiped anyway
            return render.DataGrid(pd.DataFrame([]))

        ta_list = cache_acquisition[target.host]['targets']
        acq_target_list.set(ta_list)
        names, G_mag, t_eff, log_g, ra, dec, separation = ta_list
        data_df = {
            'Gaia DR3 target': [name[9:] for name in names],
            'G_mag': [f'{mag:5.2f}' for mag in G_mag],
            'separation (")': [f'{sep:.3f}' for sep in separation],
            'T_eff (K)': [f'{temp:.1f}' for temp in t_eff],
            'log(g)': [f'{grav:.2f}' for grav in log_g],
            'RA (deg)': [f'{r:.4f}' for r in ra],
            'dec (deg)': [f'{d:.4f}' for d in dec],
        }
        acquisition_df = pd.DataFrame(data=data_df)
        return render.DataGrid(
            acquisition_df,
            selection_mode="row",
            height='370px',
            summary=True,
        )


    @reactive.effect
    def select_ta_row():
        """
        Gets triggrered all the time. Can I limit it to true user-interactions
        with acquisition_targets
        """
        acq_target_list.get()
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)

        if target is None or target.host not in cache_acquisition:
            return
        target_list = cache_acquisition[target.host]['targets']

        # TBD: if acquisition_targets changed, a previous non-zero cell_selection()
        # will leak into the new dataframe, which will de-synchronize
        # cache_acquisition[target.host]['selected']
        df = acquisition_targets.cell_selection()
        if df is None or len(df['rows'])==0:
            return

        df_data = acquisition_targets.data()
        current_data = [f'Gaia DR3 {id}' for id in df_data["Gaia DR3 target"]]
        if current_data[0] != target_list[0][0]:
            acquisition_targets._reset_reactives()
            return

        cache_acquisition[target.host]['selected'] = idx = df['rows'][0]
        target_name = target_list[0][idx]

        deselect_targets = {'event': 'deselectAllShapes'}
        select_acq_target = {
            'event': 'selectShape',
            'content': {
                'overlayName': 'Nearby Gaia sources',
                'shapeName': target_name
            }
        }
        esasky_command.set([deselect_targets, select_acq_target])


    # TBD: rename
    @reactive.effect
    @reactive.event(input.get_acquisition_target)
    def _():
        name = input.target.get()
        target = catalog.get_target(name, is_transit=None, is_confirmed=None)
        if target is None:
            return
        if target.host not in cache_acquisition:
            error_msg = ui.markdown(
                "First click the '*Search nearby targets*' button, then select "
                "a target from the '*Acquisition targets*' tab"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            return

        selected = acquisition_targets.cell_selection()['rows']
        if len(selected) == 0:
            error_msg = ui.markdown(
                "First select a target from the '*FOV targets*' tab"
            )
            ui.notification_show(error_msg, type="warning", duration=5)
            return
        target_list = cache_acquisition[target.host]['targets']
        names, G_mag, t_eff, log_g, ra, dec, separation = target_list
        idx = selected[0]
        text = (
            f"\nacq_target = {repr(names[idx])}\n"
            f"gaia_mag = {G_mag[idx]}\n"
            f"separation = {separation[idx]}\n"
            f"t_eff = {t_eff[idx]}\n"
            f"log_g = {log_g[idx]}\n"
            f"ra = {ra[idx]}\n"
            f"dec = {dec[idx]}"
        )
        print(text)


app = App(app_ui, server)
