# Copyright (c) 2025 Patricio Cubillos and A. G. Sreejith
# WALTzER is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import pyratbay.spectrum as ps
from pyratbay.tools import u
import plotly.graph_objects as go
import plotly.express as px

__all__ = [
    'bin_spectrum',
    'plotly_variances',
    'plotly_flux_snr',
    'plotly_depth_snr',
    'plotly_tso_spectra',
]


COLOR_SEQUENCE = [
    'Royalblue', # blue
    '#15b01a',  # green
    '#000075',  # navy
    '#f032e6',  # magenta
    '#42d4f4',  # cyan
    '#888888',  # grey
    'Red',      # red
    '#9A6324',  # brown
    '#800000',  # maroon
    '#000000',  #  black
    '#469990',  # teal
    '#911eb4',  # purple
    '#808000',  # olive
    'Green',  # green
]
blue = '#4169E1'
green = '#15B01A'
tomato = '#FF6347'
obs_col = px.colors.sample_colorscale('Viridis', 0.2)[0]
model_col = px.colors.sample_colorscale('Viridis', 0.75)[0]
            
depth_units_label = {
    'none': '', 
    'percent': ' (%)',
    'ppm': ' (ppm)',
}


def bin_spectrum(wl, var_list, resolution=None, binsize=1):
    """
    Bin variance list
    """
    nvars = len(var_list)
    if resolution == 0:
        return wl, var_list

    if resolution is None:
        if binsize == 1:
            return wl, var_list

        nwave = len(wl)
        bin_idx = np.arange(0, nwave, binsize)
        counts = np.diff(np.append(bin_idx, nwave))
        bin_wl = np.add.reduceat(wl, bin_idx) / counts

        bin_spectrum = [
            np.add.reduceat(var_list[j], bin_idx)
            for j in range(nvars)
        ]
        return bin_wl, bin_spectrum

    wl_min = np.amin(wl)
    wl_max = np.amax(wl)
    bin_edges = ps.constant_resolution_spectrum(
        wl_min, wl_max, resolution,
    )
    bin_edges = np.append(bin_edges, wl_max)
    bin_wl = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    nbins = len(bin_wl)
    bin_spectrum = np.zeros((nvars,nbins))
    for i in range(nbins):
        bin_mask = (wl>=bin_edges[i]) & (wl<bin_edges[i+1])
        for j in range(nvars):
            bin_spectrum[j,i] = np.sum(var_list[j][bin_mask])

    return bin_wl, bin_spectrum


def plotly_variances(
        tso,
        wl_range=None,
        wl_scale='linear',
        binsize=None,
    ):
    """
    Make a plotly figure of the variances.
    """
    fig = go.Figure(
        layout={'colorway': COLOR_SEQUENCE},
    )

    labels = ['source', 'sky', 'dark', 'read noise']
    colors = [blue,'tomato', green, 'black']

    bands = tso['meta']['bands']
    print(bands)
    wl_min = 2.0
    wl_max = 0.0
    for j,band in enumerate(bands):
        var = tso[band]
        band_type = var['det_type']
        show_legend = True if j==0 else False

        wl = var['wl']
        wl_min = np.amin([wl_min, 0.95*var['wl_min']])
        wl_max = np.amax([wl_max, 1.05*var['wl_max']])

        if band_type=="photometry":
            marker = dict(symbol="circle", size=5)
            error = 0.5*(var['wl_max']-var['wl_min'])
            error_x = dict(
                type='data', visible=True,
                array=[error],
            )
            variances = var['variances']
        else:
            marker = dict()
            error_x = None
            wl, variances = bin_spectrum(wl, var['variances'], binsize=binsize)
        line = dict(shape='linear')


        for i in range(len(labels)):
            fig.add_trace(go.Scatter(
                x=wl,
                y=variances[i],
                error_x=error_x,
                mode="lines+markers" if marker else "lines",
                marker=marker,
                line=line,
                name=f'{labels[i]}',
                legendgroup=labels[i],
                showlegend=show_legend,
                line_color=colors[i],
            ))

    fig.update_traces(
        hovertemplate=
            'wl = %{x:.2f}<br>'+
            'flux = %{y:.3f}'
    )
    fig.update_yaxes(
        title_text='electrons / second',
        title_standoff=0,
        type='log',
    )

    wl_range = np.array([wl_min, wl_max])
    if wl_scale == 'log':
        wl_range = np.log10(wl_range)

    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )

    fig.update_layout(legend=dict(
        tracegroupgap=0,
        yanchor="bottom",
        xanchor="right",
        y=0.01,
        x=0.99,
    ))
    fig.update_layout(showlegend=True)
    return fig


def plotly_flux_snr(
        flux_data,
        wl_scale='linear',
        y_scale='linear',
        y_label='',
    ):
    """
    Make a plotly figure of stellar S/N spectra.
    """
    fig = go.Figure(
        layout={'colorway': COLOR_SEQUENCE},
    )

    bands = flux_data[0]
    for j,band in enumerate(bands):
        wl = flux_data[1][j]
        snr_in = flux_data[2][j] / np.sqrt(flux_data[4][j])
        snr_out = flux_data[3][j] / np.sqrt(flux_data[5][j])
        show_legend = j == 0

        if band=="nir":
            marker = dict(symbol="circle", size=5)
            error = flux_data[6][j]
            error_x = dict(
                type='data', visible=True,
                array=error,
            )
        else:
            marker = dict()
            error_x = None
        line = dict(shape='linear')


        fig.add_trace(go.Scatter(
            x=wl,
            y=snr_in,
            error_x=error_x,
            mode="lines+markers" if marker else "lines",
            marker=marker,
            line=line,
            name=f'in-transit {band}',
            legendgroup='in-transit',
            showlegend=show_legend,
            line_color=blue,
        ))

        fig.add_trace(go.Scatter(
            x=wl,
            y=snr_out,
            error_x=error_x,
            mode="lines+markers" if marker else "lines",
            marker=marker,
            line=line,
            name=f'out-of-transit {band}',
            legendgroup='out-of-transit',
            showlegend=show_legend,
            line_color=tomato,
        ))

    fig.update_traces(
        hovertemplate=
            'wl = %{x:.4f}<br>'+
            'SNR = %{y:.1f}'
    )
    fig.update_yaxes(
        title_text=y_label,
        title_standoff=0,
        type=y_scale,
    )

    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        type=wl_scale,
    )

    fig.update_layout(legend=dict(
        tracegroupgap=0,
        yanchor="bottom",
        xanchor="right",
        y=0.01,
        x=0.99,
    ))
    fig.update_layout(showlegend=True)
    return fig


def plotly_depth_snr(
        tso,
        depth_data,
        wl_range=None,
        wl_scale='linear',
        y_scale='linear',
        y_label='',
    ):
    """
    Make a plotly figure of stellar S/N spectra.
    """
    if wl_range is None:
        wl_range = [None, None]
    fig = go.Figure(
        layout={'colorway': COLOR_SEQUENCE},
    )

    bands = depth_data[0]
    wl_min = np.amin([tso[band]['wl_min'] for band in bands]) * 0.9
    wl_max = np.amax([tso[band]['wl_max'] for band in bands]) * 1.1
    if wl_range[0] is None:
        wl_range[0] = wl_min
    if wl_range[1] is None:
        wl_range[1] = wl_max

    if wl_scale == 'log':
        wl_range = [np.log10(val) for val in wl_range]

    for j,band in enumerate(bands):
        wl = depth_data[1][j]
        snr_in = depth_data[2][j] / np.sqrt(depth_data[3][j])
        show_legend = j == 0

        if band=="nir":
            marker = dict(symbol="circle", size=5)
            error = depth_data[4][j]
            error_x = dict(
                type='data', visible=True,
                array=error,
            )
        else:
            marker = dict()
            error_x = None
        line = dict(shape='linear')


        fig.add_trace(go.Scatter(
            x=wl,
            y=snr_in,
            error_x=error_x,
            mode="lines+markers" if marker else "lines",
            marker=marker,
            line=line,
            name=f'{band}',
            showlegend=show_legend,
            line_color=obs_col,
        ))

    fig.update_traces(
        hovertemplate=
            'wl = %{x:.4f}<br>'+
            'SNR = %{y:.1f}'
    )
    fig.update_yaxes(
        title_text=y_label,
        title_standoff=0,
        type=y_scale,
    )

    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )

    fig.update_layout(legend=dict(
        tracegroupgap=0,
        yanchor="bottom",
        xanchor="right",
        y=0.01,
        x=0.99,
    ))
    fig.update_layout(showlegend=True)
    return fig


def plotly_tso_spectra(
        tso, tso_data, depth_model, resolution=250.0,
        model_label='model', instrument_label=None,
        wl_range=None, wl_scale='linear',
        depth_units='percent', depth_range=None,
        obs_geometry='transit',
    ):
    """
    Make a plotly figure of transit/eclipse depth TSO spectra.
    """
    bands, bin_wl, bin_depth, bin_err, bin_widths = tso_data
    d_units = u(depth_units)
    fig = go.Figure()

    ymax = 0.0
    ymin = np.inf

    wl_min = np.amin([tso[band]['wl_min'] for band in bands]) * 0.9
    wl_max = np.amax([tso[band]['wl_max'] for band in bands]) * 1.1
    if wl_range[0] is None:
        wl_range[0] = wl_min
    if wl_range[1] is None:
        wl_range[1] = wl_max

    # The simulations
    for i,band in enumerate(bands):
        error = bin_err[i]/d_units
        error_y = dict(type='data', array=error, visible=True, width=0)
        if band=="nir":
            error_x = dict(type='data', array=bin_widths[i], visible=True)
        else:
            error_x = None


        marker = dict(symbol="circle", color=obs_col, size=5)
        show_legend = True

        fig.add_trace(go.Scatter(
            x=bin_wl[i],
            y=bin_depth[i]/d_units,
            error_x=error_x,
            error_y=error_y,
            mode='markers',
            marker=marker,
            name=f'WALTzER {band}',
            #legendgroup=instrument_label[i],
            showlegend=show_legend,
        ))
        ymax = np.amax([ymax, np.amax(bin_depth[i])])
        ymin = np.amin([ymin, np.amin(bin_depth[i])])

    # The model
    wl, depth = depth_model
    fig.add_trace(go.Scatter(
        x=wl,
        y=depth/u(depth_units),
        mode='lines',
        name='model_label',
        line=dict(color=model_col, width=1.5),
        opacity=0.85,
    ))

    fig.update_traces(
        hovertemplate=
            'wl = %{x:.2f}<br>'+
            'depth = %{y:.3f}'
    )
    if depth_range is None:
        ymax = ymax/d_units
        ymin = ymin/d_units
        dy = 0.1 * (ymax-ymin)
        depth_range = [ymin-dy, ymax+dy]
    ylabel = f'{obs_geometry} depth{depth_units_label[depth_units]}'
    fig.update_yaxes(
        title_text=ylabel,
        title_standoff=0,
        range=depth_range,
    )

    if wl_scale == 'log':
        if wl_range[0] is not None:
            wl_range[0] = np.log10(wl_range[0])
        if wl_range[1] is not None:
            wl_range[1] = np.log10(wl_range[1])

    fig.update_xaxes(
        title_text='wavelength (um)',
        title_standoff=0,
        range=wl_range,
        type=wl_scale,
    )

    fig.update_layout(legend=dict(
        orientation="h",
        entrywidth=1.0,
        entrywidthmode='fraction',
        yanchor="bottom",
        xanchor="right",
        y=1.02,
        x=1
    ))
    fig.update_layout(showlegend=True)
    return fig

