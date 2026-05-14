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

from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

import numpy as np

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
    credible_interval: float = 0.90,
    chain_labels: Optional[str | list[str]] = None,
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
    fast_kde: bool = True,
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
    chain_labels : str or list[str], optional
        Label(s) for the chain(s) to be used in the legend. If a single string is provided and multiple chains are given, the same label will be used for all chains.
    colors : list[str], optional
        List of colors for the chains. If not provided, default colors will be used.
    contour_levels : tuple[float, ...], default (0.68, 0.90)
        Levels for the 2D contour plots, specified as fractions of the total probability mass (e.g., 0.68 for 68% credible region).
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
        Keyword arguments for the KDE estimation.
    offdiag_kwargs : dict, optional
        Keyword arguments for the off-diagonal plots (histogram, hexbin, contour).
    truth_kwargs : dict, optional
        Keyword arguments for the truth lines.
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    fast_kde : bool, default True
        If True, use the `JAX`-based KDE estimation of :mod:`KDExpress`. If False, use the :mod:`scipy` gaussian KDE estimation.
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

        if colors is not None and len(colors) != num_chains:
            raise ValueError("Number of colors does not match the number of chains")
        if colors is None:
            colors = DEFAULT_COLORLIST[:num_chains]

        if chain_labels is not None and len(chain_labels) != num_chains:
            raise ValueError("Number of labels does not match the number of chains")

        if plot_delta and truths is None:
            raise ValueError(
                "A dictionary of true values is required if `plot_delta` is True"
            )

        _truths = truths.copy() if truths is not None else {}
        all_columns = list(samples[0].keys())

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
                    tmp_chain[new_key] = chain[old_key] - _truths[old_key]
                _chains.append(tmp_chain)
            columns = new_columns
            truths = {key: 0.0 for key in columns}
        else:
            for chain in samples:
                tmp_chain = {}
                for key in columns:
                    tmp_chain[key] = chain[key]

                _chains.append(tmp_chain)

        # ── defaults ──────────────────────────────────────────────────────────────

        kde_kwargs = dict(kde_kwargs or {})
        offdiag_kwargs = dict(offdiag_kwargs or {})
        truth_kwargs = dict(truth_kwargs or {})
        truth_kwargs = {"color": "k", "ls": "--", "lw": 1.2, **truth_kwargs}

        kde_bw = kde_kwargs.get("bandwidth", "scott")

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

        # ── diagonal: 1D KDE ──────────────────────────────────────────────────────
        for i in range(n_dim):
            ax = axes[i, i]
            for c_idx, (chain_here, color) in enumerate(zip(_chains, colors)):

                data = chain_here[columns[i]]
                w = _weights[c_idx]

                x, pdf = kde_1d(data, bw=kde_bw, fast=fast_kde, weights=w)
                lo, med, hi = get_credible_interval(data, credible_interval, weights=w)

                ax.plot(x, pdf, **{"color": color, **kde_kwargs})
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
            if i < n_dim - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(columns[i], fontsize=label_fontsize)

                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))

                ax.xaxis.labelpad = xlabelpad

            ax.tick_params(axis="x", labelsize=tick_labelsize)

        # ── off-diagonal: configurable 2D density ─────────────────────────────────
        for i in range(1, n_dim):
            for j in range(i):
                ax = axes[i, j]

                for c_idx, (chain_here, color) in enumerate(zip(_chains, colors)):
                    xd, yd = chain_here[columns[j]], chain_here[columns[i]]
                    w = _weights[c_idx]
                    cmap = chain_cmap(color)

                    offdiag_hist_fn(ax, xd, yd, w, cmap, **offdiag_kwargs)

                    if _use_kde:
                        x_out, y_out, z_out = kde_2d(
                            xd, yd, bw=kde_bw, fast=fast_kde, weights=w
                        )
                        lvls = hdi_levels(z_out, contour_levels)

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

                    ax.xaxis.labelpad = xlabelpad

                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(columns[i], fontsize=label_fontsize)

                    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))

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

            max_handles_per_column = 4

            _legend_kwargs = {
                "fontsize": scale_font(plt.rcParams["legend.fontsize"], num_dim=n_dim),
                "markerscale": max(0.6, 1.2 - 0.05 * n_dim),
                "frameon": False,
                "fancybox": True,
                "title_fontsize": label_fontsize,
                "ncol": round(len(handles) / max_handles_per_column),
            }

            _legend_kwargs.update(legend_kwargs or {})

            # Place the legend next to the first diagonal panels
            fig.legend(handles=handles, loc="upper center", **_legend_kwargs)

            # find where the diagonal panels end and move the legend there
            reposition_legend(fig, n_dim)

        fig.align_labels()
        fig.tight_layout()
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        if savepath is not None:
            plt.savefig(savepath)
        return fig, axes
