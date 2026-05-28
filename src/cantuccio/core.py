"""
core.py
=======

Core plotting logic.

corner(samples, ...)  ->  (fig, axes)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

from .kde import hdi_levels, kde_1d, kde_2d
from .visuals import (
    DEFAULT_COLORLIST,
    chain_cmap,
    get_stylefile,
    reposition_legend,
    scale_font,
)

OFFDIAG_MODES = {
    "hist",
    "hexbin",
    "contour",
    "kde",
    "hist+kde",
    "hexbin+kde",
    "contour+kde",
}


def _hist_fn(ax, x, y, weights, cmap, **kwargs):
    ax.hist2d(
        x,
        y,
        weights=weights,
        **{"bins": 20, "density": True, "cmap": cmap, **kwargs},
    )


def _hexbin_fn(ax, x, y, weights, cmap, **kwargs):
    ax.hexbin(
        x,
        y,
        C=weights,
        reduce_C_function=np.sum if weights is not None else None,
        **{"gridsize": 15, "mincnt": 1, "cmap": cmap, "lw": 0.0, **kwargs},
    )


def _pass_fn(*args, **kwargs):
    pass

def get_credible_interval(
    data: np.ndarray, level: float, weights: np.ndarray | None = None
) -> tuple[float, float, float]:
    """
    Return (lower, median, upper) for a highest-density credible interval.
    Standalone function for users who want to compute credible intervals without going through the kde estimation. This is not used internally by the :meth:`cornerplot` method.

    Parameters
    ----------
    data : np.ndarray
        1D array of samples.
    level : float
        Credible interval level, e.g. 0.90 for a 90% credible interval.
    weights : np.ndarray, optional
        1D array of weights corresponding to the samples.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the lower bound, median, and upper bound of the credible interval.
    """
    lo = 100 * (1.0 - level) / 2.0
    percentiles = [lo, 50.0, 100.0 - lo]
    if weights is None:
        return tuple(np.percentile(data, percentiles))

    i = np.argsort(data)
    d = data[i]
    w = weights[i]
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= np.sum(w)
    return tuple(np.interp(np.array(percentiles) / 100.0, cdf, d))


def get_credible_interval_median(
    x: np.ndarray, pdf: np.ndarray, level: float,
) -> tuple[float, float, float]:
    """
    Return (lower, median, upper). The median is the value of x at which the cumulative distribution function (CDF) reaches 0.5, and the lower and upper bounds are the values of x at which the CDF reaches (1 - level) / 2 and 1 - (1 - level) / 2, respectively.

    Parameters
    ----------
    x : np.ndarray
        1D array of samples.
    pdf : np.ndarray
        1D array of probability density values corresponding to the samples.
    level : float
        Credible interval level, e.g. 0.90 for a 90% credible interval.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the lower bound, median, and upper bound of the credible interval.
    """
    assert level < 1.0, "Credible interval level must be less than 1.0"
    vals = [0.5 - level / 2, 0.5, 0.5 + level / 2]
    cdf = pdf.cumsum()
    cdf /= cdf.max()  # Normalize to ensure the cumulative distribution goes from 0 to 1
    bounds = interp1d(cdf, x)(vals)
    bounds[1] = 0.5 * (bounds[0] + bounds[2])
    return tuple(bounds)

def get_credible_interval_hdi(
        x: np.ndarray, pdf: np.ndarray, level: float
) -> tuple[float, float, float]:
    """
    Return (lower, center, upper) for the highest-density credible interval.

    Routine adapted from the `ChainConsumer` package: https://samreay.github.io/ChainConsumer/, doi:10.21105/joss.00045.

    Parameters
    ----------
    x : np.ndarray
        1D array of samples.
    pdf : np.ndarray
        1D array of probability density values corresponding to the samples.
    level : float
        Credible interval level, e.g. 0.90 for a 90% credible interval.
    
    Returns
    -------
    tuple[float, float, float]
        A tuple containing the lower bound, center, and upper bound of the highest-density interval.
    """
    cdf = pdf.cumsum()
    cdf /= cdf.max()  # Normalize to ensure the cumulative distribution goes from 0 to 1

    x_in = np.concatenate([[x[0]], x])
    cdf_in = np.concatenate([[0.0], cdf])

    eps = 1e-12
    best_width = float("inf")
    best_lower = float(x_in[0])
    best_upper = float(x_in[-1])
    best_start_mass = 0.0
    best_end_mass = 1.0

    for start_idx, start_mass in enumerate(cdf_in[:-1]):
        target = start_mass + level
        if target > 1.0 + eps:
            break

        end_idx = np.searchsorted(cdf_in, target, side="left")

        # Ensure at least one point is in the interval
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        if end_idx >= cdf_in.size:
            break

        # If still slightly under target, move one step right if possible
        if cdf_in[end_idx] - start_mass < level - eps and end_idx + 1 < cdf_in.size:
            end_idx += 1

        lower = float(x_in[start_idx])
        upper = float(x_in[end_idx])
        width = upper - lower
        if width <= eps:
            continue

        if width < best_width - eps:
            best_width = width
            best_lower = lower
            best_upper = upper
            best_start_mass = float(start_mass)
            best_end_mass = float(cdf_in[end_idx])

    interval_mass = best_end_mass - best_start_mass

    if interval_mass <= eps:
        center = 0.5 * (best_lower + best_upper)

    else:
        center_mass = best_start_mass + 0.5 * interval_mass
        center = float(np.interp(center_mass, cdf_in, x_in, left=best_lower, right=best_upper))

    return best_lower, center, best_upper

_CREDIBLE_INTERVAL_REGISTRY = {
    "median": get_credible_interval_median,
    "hdi": get_credible_interval_hdi,
}

def overplot_lines(
    axes: np.ndarray,
    lines: dict[str, float],
    columns: Optional[list[str]] = None,
    marker: Optional[str] = None,
    **kwargs,
):
    """
    Overplot vertical and horizontal lines on a corner plot.

    Parameters
    ----------
    axes : np.ndarray
        Array of matplotlib axes objects.
    lines : dict[str, float]
        Dictionary mapping parameter names to line positions.
    columns : list[str], optional
        List of parameter names. If not provided, the keys of the `lines` dictionary will be used.
    marker : str, optional
        Marker style for the line plots. If not provided, lines will be plotted without markers.
    **kwargs
        Additional keyword arguments for the line plots.
    """
    if columns is None:
        columns = list(lines.keys())

    n_dim = len(columns)

    assert axes.shape == (
        n_dim,
        n_dim,
    ), "Axes array shape must match the number of columns"

    x_here = False
    y_here = False

    for i in range(n_dim):
        for j in range(i + 1):
            ax = axes[i, j]
            if j == i:
                # Diagonal: vertical line
                if columns[i] in lines and lines[columns[i]] is not None:
                    ax.axvline(lines[columns[i]], **kwargs)
            else:
                # Off-diagonal: horizontal and vertical lines
                if columns[j] in lines and lines[columns[j]] is not None:
                    x = lines[columns[j]]
                    ax.axvline(x, **kwargs)
                    x_here = True
                if columns[i] in lines and lines[columns[i]] is not None:
                    y = lines[columns[i]]
                    ax.axhline(y, **kwargs)
                    y_here = True

                if marker is not None and x_here and y_here:
                    ax.plot(x, y, marker=marker, **kwargs)

                x_here = False
                y_here = False


def cornerplot(  # pylint: disable=too-many-branches, too-many-statements too-many-arguments too-many-positional-arguments too-many-locals
    samples: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    columns: list[str] | None = None,
    weights: np.ndarray | list[np.ndarray] | None = None,
    truths: Optional[dict] = None,
    plot_delta: bool = False,
    periodic: Optional[dict[str, tuple[float, float]]] = None,
    credible_interval: float = 0.90,
    statistic: str = "median",
    labels: Optional[str | list[str]] = None,
    colors: Optional[list[str]] = None,
    contour_levels: tuple[float, ...] = (0.68, 0.90),
    title_format: Optional[str] = None,
    fig: Optional[Figure] = None,
    axes: Optional[np.ndarray] = None,
    offdiag_mode: str = "hexbin+kde",
    kde_kwargs: Optional[dict] = None,
    offdiag_kwargs: Optional[dict] = None,
    truth_kwargs: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
    n_ticks: int = 4,
    xlabelpad: float | None = 4.0,
    ylabelpad: float | None = 2.0,
    hspace=0.1,
    wspace=0.1,
    stylefile: str | None = None,
    savepath: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Custom corner plot with KDE marginals and 2D contours/hexbin/histograms.

    Parameters
    ----------
    samples : dict[str, np.ndarray] or list[dict[str, np.ndarray]]
        A single chain (dict) or a list of chains (list of dicts), where each dict maps parameter names to samples.
    columns : list[str], optional
        List of parameter names to include in the plot. If None, all keys from the first chain will be used.
    weights : np.ndarray or list[np.ndarray], optional
        A single array of weights or a list of arrays of weights corresponding to the chains. If None, samples will be treated as unweighted.
    truths : dict, optional
        Dictionary mapping parameter names to their true values. If provided, these will be overplotted as lines on the plot.
    plot_delta : bool, default False
        If True, plot the difference between the samples and the truths (i.e., Δ = samples - truths) instead of the raw samples. Requires `truths` to be provided.
    credible_interval : float, default 0.90
        Credible interval level for shading the 1D KDE plots on the diagonal.
    statistic : str, default "median"
        Statistic to compute for the titles on the diagonal panels. Options are "median" or "hdi". The latter follows the `ChainConsumer` implementation. Refer to the :meth:`get_credible_interval_median` and :meth:`get_credible_interval_hdi` for details.
    labels : str or list[str], optional
        Label(s) for the chain(s) to be used in the legend. If a single string is provided and multiple chains are given, the same label will be used for all chains.
    colors : list[str], optional
        List of colors for the chains. If not provided, default colors will be used.
    contour_levels : tuple[float, ...], default (0.68, 0.90)
        Levels for the 2D contour plots, specified as fractions of the total probability mass (e.g., 0.68 for 68% credible region).
    periodic : dict[str, tuple[float, float]], optional
        Dictionary mapping parameter names to their wrapped domain boundaries (e.g., {"phi": (0, 2 * np.pi)} for an angle parameter).
        If provided, the KDE contours and 1D marginal regions correctly integrate mathematically over boundaries to fold and wrap inside this topology.
        We recommend providing the periodic bounds only for parameters that wrap within the range of the samples. 
    title_format : str, optional
        Format string for the titles on the diagonal panels. If provided, the median and credible interval will be included in the title for each parameter. The format string should be suitable for formatting the median and interval widths, e.g. ".2f" for 2 decimal places.
    fig : matplotlib.figure.Figure, optional
        Matplotlib Figure object to use for the plot. If None, a new figure will be created.
    axes : np.ndarray, optional
        Array of matplotlib Axes objects to use for the plot. If None, a new set of axes will be created. The shape of the axes array should match the number of parameters (n_dim x n_dim).
    offdiag_mode : str, default "hexbin+kde"
        Mode for the off-diagonal panels. Options are:
        - "hist": 2D histogram
        - "hexbin": Hexagonal binning
        - "contour": Filled contour plot of the KDE
        - "kde": Non-filled contour plot of the KDE
        - "hist+kde": 2D histogram with KDE contours overlaid
        - "hexbin+kde": Hexagonal binning with KDE contours overlaid
        - "contour+kde": Filled contour plot with KDE contours overlaid
    kde_kwargs : dict, optional
        Keyword arguments for the KDE estimation. If not provided, defaults will be used (e.g., bandwidth method "silverman" and FFT-based KDE). Supported keys include:
        - "bandwidth": Bandwidth method for KDE ("scott", "silverman", or a scalar factor).
        - "fast": If True, use FFT-based KDE through the `KDEpy` package for faster computation on large datasets. If False, use the standard `scipy.stats.gaussian_kde`. Default is False.
        - "num_1d": Number of points to evaluate the 1D KDE on. Default is 512.
        - "num_2d": Number of points per dimension to evaluate the 2D KDE on. Default is 80.
    offdiag_kwargs : dict, optional
        Keyword arguments for the off-diagonal plots (histogram, hexbin, contour).
    truth_kwargs : dict, optional
        Keyword arguments for the truth lines.
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    n_ticks : int, default 4
        Number of ticks to show on each axis.
    xlabelpad : float, optional
        Padding for x-axis labels. If None, the default Matplotlib padding will be used.
    ylabelpad : float, optional
        Padding for y-axis labels. If None, the default Matplotlib padding will be used.
    hspace : float, default 0.1
        Height space between subplots.
    wspace : float, default 0.1
        Width space between subplots.
    stylefile : str, optional
        Path to a Matplotlib style file to use for the plot. If None, a default style file included with the package will be used.
    savepath : str, optional
        Path to save the figure. If None, the figure will not be saved.
    """

    # ———— Blind unwanted columns, subtract truth if required.

    if stylefile is None:
        stylefile = get_stylefile()

    with plt.style.context(stylefile):
        
        chain_labels = labels # give a more descriptive name

        if not isinstance(samples, list):
            samples = [samples]
            if isinstance(chain_labels, str):
                chain_labels = [chain_labels]
            if weights is not None and not isinstance(weights, list):
                weights = [weights]

        num_chains = len(samples)

        if weights is not None and len(weights) != num_chains:
            raise ValueError(
                "Number of weight arrays does not match the number of chains"
            )
        _weights = weights if weights is not None else [None] * num_chains

        if colors is not None:
            if isinstance(colors, str):
                colors = [colors]
            if len(colors) != num_chains:
                raise ValueError("Number of colors does not match the number of chains")
        else:
            colors = DEFAULT_COLORLIST[:num_chains]

        if chain_labels is not None and len(chain_labels) != num_chains:
            raise ValueError("Number of labels does not match the number of chains")
            
        # We no longer manually unwrap the data here. The user passes the data wrapped or unwrapped, 
        # and KDE evaluates and folds correctly onto [low, high] bounds based on `periodic`.

        if plot_delta and truths is None:
            raise ValueError(
                "A dictionary of true values is required if `plot_delta` is True"
            )

        _truths = truths.copy() if truths is not None else {}
        seen = set()
        all_columns = []
        for chain in samples:
            for k in chain.keys():
                if k not in seen:
                    seen.add(k)
                    all_columns.append(k)

        if columns is None:
            columns = all_columns

        n_dim = len(columns)

        _chains = []
        if plot_delta:

            new_columns = []
            for key in columns:
                if key.startswith("$"):
                    new_key = r"$\Delta\," + key[1:]
                else:
                    new_key = rf"$\Delta\,${key}"

                new_columns.append(new_key)

            for chain in samples:
                tmp_chain = {}
                for old_key, new_key in zip(columns, new_columns):
                    if old_key in chain:
                        tmp_chain[new_key] = chain[old_key] - _truths[old_key]
                _chains.append(tmp_chain)
            columns = new_columns
            truths = {key: 0.0 for key in columns}
        else:
            for chain in samples:
                tmp_chain = {}
                for key in columns:
                    if key in chain:
                        tmp_chain[key] = chain[key]
                _chains.append(tmp_chain)

        # ── defaults ──────────────────────────────────────────────────────────────

        kde_kwargs = dict(kde_kwargs or {})
        offdiag_kwargs = dict(offdiag_kwargs or {})
        truth_kwargs = dict(truth_kwargs or {})
        truth_kwargs = {"color": "k", "ls": "--", "lw": 1.2, **truth_kwargs}

        kde_bw = kde_kwargs.pop("bandwidth", "silverman")
        kde_fast = kde_kwargs.pop("fast", True)
        kde_num_1d = kde_kwargs.pop("num_1d", 512)
        kde_num_2d = kde_kwargs.pop("num_2d", 80)

        if offdiag_mode not in OFFDIAG_MODES:
            raise ValueError(
                f"offdiag_mode {offdiag_mode!r} not recognised. " f"Use {OFFDIAG_MODES}"
            )

        if len(samples) > 5 and offdiag_mode != "kde":
            offdiag_mode = "kde"
            warnings.warn(
                f"Too many chains ({len(samples)} >5), setting offdiag_mode to 'kde'",
                UserWarning,
            )

        if "+" in offdiag_mode:
            base_mode, overlay_mode = offdiag_mode.split("+")
        else:
            base_mode = offdiag_mode
            overlay_mode = None

        if base_mode == "hist":
            offdiag_hist_fn = _hist_fn
        elif base_mode == "hexbin":
            offdiag_hist_fn = _hexbin_fn
        else:
            offdiag_hist_fn = _pass_fn

        _use_kde = base_mode in ["contour", "kde"] or overlay_mode == "kde"
        
        if statistic not in _CREDIBLE_INTERVAL_REGISTRY:
            raise ValueError(f"Invalid statistic {statistic!r}. Supported options are: {list(_CREDIBLE_INTERVAL_REGISTRY.keys())}")

        # ── figure / axes / fontsizes ──────────────────────────────────────────
        if fig is None or axes is None:
            figsize = (2.0 * n_dim, 2.0 * n_dim)
            fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize, squeeze=False)

        label_fontsize = scale_font(plt.rcParams["axes.labelsize"], n_dim)
        tick_labelsize = scale_font(plt.rcParams["xtick.labelsize"], n_dim)

        # Hide upper triangle
        for i in range(n_dim):
            for j in range(i + 1, n_dim):
                axes[i, j].set_visible(False)

        all_left_limit = {k: float("inf") for k in columns}
        all_right_limit = {k: float("-inf") for k in columns}

        # ── diagonal: 1D KDE ──────────────────────────────────────────────────────
        for i in range(n_dim):
            ax = axes[i, i]
            for c_idx, (chain_here, color) in enumerate(zip(_chains, colors)):

                data = chain_here.get(columns[i])
                if data is None:
                    continue
                w = _weights[c_idx]

                periodic_bnds = periodic.get(columns[i]) if periodic else None
                x, pdf = kde_1d(data, bw=kde_bw, weights=w, fast=kde_fast, n=kde_num_1d, periodic=periodic_bnds)

                lo, med, hi = _CREDIBLE_INTERVAL_REGISTRY[statistic](x, pdf, credible_interval)

                ax.plot(x, pdf, **{"color": color, **kde_kwargs})
                
                if periodic_bnds is not None:
                    data_limits = (data.min(), data.max())
                    # If the data is well within the periodic bounds, set the x-limits to the data range for better visualization. Otherwise, set the x-limits to the periodic bounds.
                    current_left = periodic_bnds[0] if data_limits[0] < periodic_bnds[0] * 1.001 else data_limits[0]
                    current_right = periodic_bnds[1] if data_limits[1] > periodic_bnds[1] * 0.999 else data_limits[1]

                    # now check if the ax has already limits set by the previous chains, and if so, take the union of the limits to ensure all chains are visible
                    all_left_limit[columns[i]] = min(all_left_limit[columns[i]], current_left)
                    all_right_limit[columns[i]] = max(all_right_limit[columns[i]], current_right)

                    ax.set_xlim(all_left_limit[columns[i]], all_right_limit[columns[i]])
                    
                    flat = np.sort(pdf)[::-1]
                    dx = np.diff(x)[0] if len(x) > 1 else 1.0
                    cumfrac = np.cumsum(flat * dx)
                    # Normalize if precision lost
                    if cumfrac[-1] > 0:
                        cumfrac /= cumfrac[-1]
                    idx = min(int(np.searchsorted(cumfrac, credible_interval)), len(flat) - 1)
                    mask = pdf >= flat[idx]
                else:
                    mask = (x >= lo) & (x <= hi)
                    
                if base_mode != "kde":
                    ax.fill_between(x, pdf, where=mask, color=color, alpha=0.50)

                if title_format is not None:
                    f = f"{{:{title_format}}}"
                    title_str = (
                        rf"{columns[i]} = ${f.format(med)}"
                        rf"_{{-{f.format(med - lo)}}}^{{+{f.format(hi - med)}}}$"
                    )
                    offset_pts = 5 + (num_chains - 1 - c_idx) * (label_fontsize * 1.2)
                    ax.annotate(
                        title_str,
                        xy=(0.5, 1.0),
                        xycoords="axes fraction",
                        xytext=(0, offset_pts),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=label_fontsize - 1.0,
                        color=color,
                    )

            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.tick_params(
                axis="y", direction=plt.rcParams["xtick.direction"]
            )  # use the style context
            
            # Lock the bottom of the y-axis exactly to 0 so the KDE rests on the floor
            ax.set_ylim(bottom=0.0)
            
            if i < n_dim - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(columns[i], fontsize=label_fontsize)

                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
                ax.xaxis.offsetText.set_fontsize(tick_labelsize)

                ax.xaxis.labelpad = xlabelpad

            ax.tick_params(axis="x", labelsize=tick_labelsize)

        # ── off-diagonal: configurable 2D density ─────────────────────────────────
        for i in range(1, n_dim):
            for j in range(i):
                ax = axes[i, j]

                for c_idx, (chain_here, color) in enumerate(zip(_chains, colors)):
                    xd = chain_here.get(columns[j])
                    yd = chain_here.get(columns[i])
                    if xd is None or yd is None:
                        continue
                    w = _weights[c_idx]
                    cmap = chain_cmap(color)

                    periodic_x = periodic.get(columns[j]) if periodic else None
                    periodic_y = periodic.get(columns[i]) if periodic else None

                    xd_plot, yd_plot = xd, yd
                    if periodic_x is not None:
                        lx, hx = periodic_x
                        xd_plot = lx + (xd - lx) % (hx - lx)
                    if periodic_y is not None:
                        ly, hy = periodic_y
                        yd_plot = ly + (yd - ly) % (hy - ly)

                    offdiag_hist_fn(ax, xd_plot, yd_plot, w, cmap, **offdiag_kwargs)

                    if _use_kde:
                        kde_num_2d_here = kde_num_2d if periodic_x is None and periodic_y is None else 800
                        x_out, y_out, z_out = kde_2d(
                            xd, yd, bw=kde_bw, weights=w, fast=kde_fast, n=kde_num_2d_here,
                            periodic_x=periodic_x, periodic_y=periodic_y
                        )
                        raw_lvls = hdi_levels(z_out, contour_levels)
                        lvls = np.unique(raw_lvls)
                        lvls = lvls[lvls < z_out.max()].tolist()

                        if len(lvls) > 0:
                            if base_mode == "contour":
                                r, g, b, _ = to_rgba(color)
                                # Power-law scaling: squaring the fraction gives 4:1 inner/outer ratio
                                band_colors = [
                                    (r, g, b, 0.50 * ((k + 1) / len(lvls)) ** 2)
                                    for k in range(len(lvls))
                                ]
                                ax.contourf(
                                    x_out,
                                    y_out,
                                    z_out,
                                    levels=[*lvls, z_out.max()],
                                    colors=band_colors,
                                )

                            if overlay_mode == "kde" or base_mode == "kde":
                                ax.contour(
                                    x_out,
                                    y_out,
                                    z_out,
                                    levels=lvls,
                                    **{"colors": [color], **offdiag_kwargs},
                                )

                if i == n_dim - 1:
                    ax.set_xlabel(columns[j], fontsize=label_fontsize)
                    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
                    ax.xaxis.offsetText.set_fontsize(tick_labelsize)

                    ax.xaxis.labelpad = xlabelpad

                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(columns[i], fontsize=label_fontsize)

                    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
                    ax.yaxis.offsetText.set_fontsize(tick_labelsize)

                    ax.yaxis.labelpad = ylabelpad
                else:
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=tick_labelsize)

        # Now add truths to the diagonal and off-diagonal panels
        if truths is not None:
            overplot_lines(axes, truths, columns=columns, **truth_kwargs)

        # ── sync axis limits and tick count ───────────────────────────────────────
        def _make_locator():
            # nbins = n_ticks-1 intervals → exactly n_ticks tick positions; no pruning
            return ticker.MaxNLocator(
                nbins=n_ticks - 1, prune=None, min_n_ticks=n_ticks
            )

        # def _make_locator():
        #     # LinearLocator strictly guarantees `numticks` but the numbers might have decimals
        #     return ticker.LinearLocator(numticks=n_ticks)

        for i in range(n_dim):
            xlim = axes[i, i].get_xlim()
            axes[i, i].xaxis.set_major_locator(_make_locator())
            axes[i, i].yaxis.set_major_locator(_make_locator())
            for row in range(i + 1, n_dim):
                axes[row, i].set_xlim(xlim)
                axes[row, i].xaxis.set_major_locator(_make_locator())
                axes[row, i].yaxis.set_major_locator(_make_locator())
            for col in range(i):
                axes[i, col].set_ylim(xlim)

        # ── optional legend ────────────────────────────────────────────────────────
        if chain_labels is not None:
            handles = [
                plt.Line2D([0], [0], color=colors[c], label=chain_labels[c])
                for c in range(num_chains)
            ]
            # add truths to the legend
            if truths is not None:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=truth_kwargs.get("color", "k"),
                        ls=truth_kwargs.get("ls", "--"),
                        label=truth_kwargs.get("label", "Truths"),
                    )
                )

            # If there's an existing legend, extract its handles and merge them
            # so we maintain a single, combined figure legend.
            if fig.legends:
                old_leg = fig.legends[0]
                existing_handles = old_leg.legend_handles
                existing_labels = [t.get_text() for t in old_leg.get_texts()]
                
                combined_handles = list(existing_handles)
                for h in handles:
                    if h.get_label() not in existing_labels:
                        combined_handles.append(h)
                handles = combined_handles
                old_leg.remove()

            max_handles_per_column = 4

            _legend_kwargs = {
                "fontsize": scale_font(plt.rcParams["legend.fontsize"], num_dim=n_dim),
                "markerscale": max(0.6, 1.2 - 0.05 * n_dim),
                "frameon": False,
                "fancybox": True,
                "title_fontsize": label_fontsize,
                "ncol": int(np.ceil(len(handles) / max_handles_per_column)),
            }

            _legend_kwargs.update(legend_kwargs or {})

            # Set the baseline anchor anchor point (upper left) so repositioning
            # aligns the left edge correctly without overlapping left subplots
            fig.legend(handles=handles, loc="upper left", **_legend_kwargs)

            # find where the diagonal panels end and move the legend there
            reposition_legend(fig, n_dim)

        fig.align_labels()
        fig.tight_layout()
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        if savepath is not None:
            plt.savefig(savepath)
        return fig, axes
