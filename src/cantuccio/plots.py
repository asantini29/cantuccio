"""
plots.py
========

Additional posterior-sample visualizations beyond the corner plot.

violinplot(samples, ...)  ->  (fig, axes)
traceplot(samples, ...)   ->  (fig, axes)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure

from .core import _CREDIBLE_INTERVAL_REGISTRY, _normalize_inputs
from .kde import kde_1d
from .visuals import DEFAULT_COLORLIST, get_stylefile, scale_font

# Vertical nudge (in row units) applied to each half's inner stats in split mode,
# so the top dataset's median/interval sits above the centerline and the bottom's below.
_STAT_OFFSET = 0.12


def _resolve_row_colors(num_chains, colors, color_by, cmap):
    """Return (per-row color list, ScalarMappable or None) for ``violinplot``."""
    if color_by is not None:
        if colors is not None:
            raise ValueError("Provide either `colors` or `color_by`, not both")
        color_by = np.asarray(color_by, dtype=float)
        if color_by.shape != (num_chains,):
            raise ValueError(
                f"color_by must have one scalar per chain ({num_chains}), got shape {color_by.shape}"
            )
        norm = Normalize(vmin=np.nanmin(color_by), vmax=np.nanmax(color_by))
        mappable = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        return [mappable.to_rgba(v) for v in color_by], mappable

    if colors is None:
        colors = [DEFAULT_COLORLIST[c % len(DEFAULT_COLORLIST)] for c in range(num_chains)]
    elif isinstance(colors, str):
        colors = [colors]
    return list(colors), None


def _draw_violin(ax, x, pdf, y_pos, scale, violin_width, color, side,
                 fill_kwargs, stats, whisker_range, show_extrema):
    """Draw one violin shape plus its inner stats on ``ax``.

    Parameters
    ----------
    side : int
        ``0`` → symmetric violin filling ``[y_pos - h, y_pos + h]`` with stats on
        the centerline (single-dataset mode). ``+1`` → top half ``[y_pos, y_pos + h]``
        with stats nudged to ``y_pos + _STAT_OFFSET``. ``-1`` → bottom half
        ``[y_pos - h, y_pos]`` with stats nudged to ``y_pos - _STAT_OFFSET``.
    scale : float
        Value to normalise ``pdf`` by (the violin reaches ``0.5 * violin_width`` at
        ``pdf == scale``).
    stats : tuple[float, float, float]
        ``(lo, med, hi)`` for the interval bar and median dot.
    whisker_range : tuple[float, float]
        ``(min, max)`` of the (periodic-wrapped) data for the whisker line.
    """
    h = 0.5 * violin_width * pdf / scale
    r, g, b, _ = to_rgba(color)

    if side == 0:
        y1, y2, stat_y = y_pos - h, y_pos + h, y_pos
    elif side > 0:
        y1, y2, stat_y = y_pos, y_pos + h, y_pos + _STAT_OFFSET
    else:
        y1, y2, stat_y = y_pos - h, y_pos, y_pos - _STAT_OFFSET

    if show_extrema:
        ax.plot(
            [whisker_range[0], whisker_range[1]], [stat_y, stat_y],
            color="k", lw=0.6, alpha=0.6, zorder=2,
        )

    ax.fill_between(
        x, y1, y2,
        **{
            "facecolor": (r, g, b, 0.9),
            "edgecolor": (r, g, b, 1.0),
            "lw": 0.8,
            "zorder": 3,
            **fill_kwargs,
        },
    )

    lo, med, hi = stats
    ax.plot(
        [lo, hi], [stat_y, stat_y],
        color="k", lw=2.5, alpha=0.8, solid_capstyle="round", zorder=4,
    )
    ax.plot(
        med, stat_y, marker="o", ms=3.5, mfc="w", mec="k", mew=0.6,
        ls="none", zorder=5,
    )


def violinplot(  # pylint: disable=too-many-branches too-many-statements too-many-arguments too-many-locals
    samples: dict[str, np.ndarray] | np.ndarray | list[dict[str, np.ndarray] | np.ndarray],
    columns: list[str] | None = None,
    weights: np.ndarray | list[np.ndarray] | None = None,
    truths: Optional[dict | np.ndarray] = None,
    plot_delta: bool = False,
    periodic: Optional[dict[str, tuple[float, float]]] = None,
    credible_interval: float = 0.90,
    statistic: str = "median",
    labels: Optional[str | list[str]] = None,
    colors: Optional[list[str]] = None,
    color_by: Optional[np.ndarray] = None,
    cmap: str = "plasma",
    colorbar_label: Optional[str] = None,
    violin_width: float = 0.8,
    show_extrema: bool = True,
    samples2: dict[str, np.ndarray] | np.ndarray | list | None = None,
    weights2: np.ndarray | list[np.ndarray] | None = None,
    split_labels: Optional[tuple[str, str] | list[str]] = None,
    split_kwargs: Optional[dict] = None,
    kde_kwargs: Optional[dict] = None,
    truth_kwargs: Optional[dict] = None,
    fig: Optional[Figure] = None,
    axes: Optional[np.ndarray] = None,
    n_ticks: int = 4,
    stylefile: str | None = None,
    savepath: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Horizontal violin plot of posterior samples: one row per chain, one panel per parameter.

    Each violin is built from the same 1D KDE machinery used by :meth:`cantuccio.core.cornerplot` (weights and periodic parameters are supported), and is annotated with a thick credible-interval bar, a white median dot, and an optional thin whisker line spanning the full sample range.

    Parameters
    ----------
    samples : dict[str, np.ndarray], np.ndarray or list[dict[str, np.ndarray] | np.ndarray]
        A single chain or a list of chains; **each chain becomes one row** of the plot. Each dict maps parameter names to samples. If `np.ndarrays` are provided instead of dictionaries, the parameters will be labelled as :math:`\\theta_i`.

        The last axis of an array is **always** the parameter/dimension axis; the dict counterpart of an array moves that axis into the keys (one key per dimension), so dict values have exactly one axis fewer:

        ============================== ============================ =====================
        Array form                     Dict counterpart (per key)   Meaning
        ============================== ============================ =====================
        ``(nsteps, ndim)``             ``(nsteps,)``                flat chain
        ``(nsteps, nwalkers, ndim)``   ``(nsteps, nwalkers)``       walker-resolved chain
        ============================== ============================ =====================

        Walker-resolved inputs are flattened over the walker axis before plotting.
    columns : list[str], optional
        List of parameter names to include in the plot. If None, the union of keys across chains will be used.
    weights : np.ndarray or list[np.ndarray], optional
        A single array of weights or a list of arrays of weights corresponding to the chains. If None, samples will be treated as unweighted.
    truths : dict or np.ndarray, optional
        Dictionary or array mapping parameter names to their true values, drawn as vertical lines on each panel. If a numpy array is provided, it will be matched to the parameters in order of `columns`.
    plot_delta : bool, default False
        If True, plot the difference between the samples and the truths (i.e., Δ = samples - truths) instead of the raw samples. Requires `truths` to be provided.
    periodic : dict[str, tuple[float, float]], optional
        Dictionary mapping parameter names to their wrapped domain boundaries (e.g., {"phi": (0, 2 * np.pi)}). The violin KDE correctly folds and wraps inside this topology.
    credible_interval : float, default 0.90
        Credible interval level for the thick bar inside each violin.
    statistic : str, default "median"
        Statistic used for the interval bar and the central dot. Options are "median" or "hdi", as in :meth:`cantuccio.core.cornerplot`.
    labels : str or list[str], optional
        Row labels for the chains (shown as y tick labels on the leftmost panel).
    colors : list[str], optional
        List of colors, one per chain/row. If not provided, the default color list is cycled. Mutually exclusive with `color_by`.
    color_by : np.ndarray, optional
        One scalar per chain/row (e.g. SNR). Violin fills are mapped through `cmap` and a colorbar is added to the figure. Mutually exclusive with `colors`.
    cmap : str, default "plasma"
        Colormap used with `color_by`.
    colorbar_label : str, optional
        Label for the colorbar created when `color_by` is given.
    violin_width : float, default 0.8
        Maximum violin height in row units (1.0 means adjacent violins touch).
    show_extrema : bool, default True
        If True, draw a thin whisker line spanning the full range of the samples behind each violin.
    kde_kwargs : dict, optional
        Keyword arguments for the KDE estimation, with the same special keys as :meth:`cantuccio.core.cornerplot` ("bandwidth", "fast", "num_1d"). Remaining entries are forwarded to ``ax.fill_between``.
    truth_kwargs : dict, optional
        Keyword arguments for the truth lines.
    fig : matplotlib.figure.Figure, optional
        Matplotlib Figure object to use for the plot. If None, a new figure will be created.
    axes : np.ndarray, optional
        Array of matplotlib Axes objects of length ``n_dim``. If None, a new set of axes will be created.
    n_ticks : int, default 4
        Number of ticks to show on each x-axis.
    stylefile : str, optional
        Path to a Matplotlib style file to use for the plot. If None, a default style file included with the package will be used.
    savepath : str, optional
        Path to save the figure. If None, the figure will not be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        The figure and the array of axes (shape ``(n_dim,)``).
    """
    if stylefile is None:
        stylefile = get_stylefile()

    if statistic not in _CREDIBLE_INTERVAL_REGISTRY:
        raise ValueError(
            f"Invalid statistic {statistic!r}. Supported options are: {list(_CREDIBLE_INTERVAL_REGISTRY.keys())}"
        )

    num_chains = len(samples) if isinstance(samples, list) else 1
    row_colors, mappable = _resolve_row_colors(num_chains, colors, color_by, cmap)

    with plt.style.context(stylefile):
        user_columns = columns  # preserve original before normalization reassigns it
        (_chains, row_colors, _weights, chain_labels, columns, truths, n_dim, num_chains
        ) = _normalize_inputs(samples, weights, row_colors, labels, user_columns, plot_delta, truths)

        split = samples2 is not None
        _chains2 = _weights2_norm = None
        if split:
            (_chains2, _, _weights2_norm, _, _, _, _, num_chains2
            ) = _normalize_inputs(samples2, weights2, None, labels, user_columns, plot_delta, truths)
            if num_chains2 != num_chains:
                raise ValueError(
                    f"samples2 must have the same number of rows (chains) as samples "
                    f"({num_chains}), got {num_chains2}"
                )
            if split_labels is not None and len(split_labels) != 2:
                raise ValueError(f"split_labels must have length 2, got {len(split_labels)}")
        _split_kwargs = {"alpha": 0.45, **dict(split_kwargs or {})}

        kde_kwargs = dict(kde_kwargs or {})
        kde_bw = kde_kwargs.pop("bandwidth", "silverman")
        kde_fast = kde_kwargs.pop("fast", True)
        kde_num_1d = kde_kwargs.pop("num_1d", 512)

        truth_kwargs = {"color": "k", "ls": "--", "lw": 1.2, **dict(truth_kwargs or {})}

        if fig is None or axes is None:
            figsize = (2.0 * n_dim, max(2.0, 0.5 * num_chains + 1.2))
            fig, axes = plt.subplots(1, n_dim, figsize=figsize, sharey=True, squeeze=False)
        axes = np.atleast_1d(np.asarray(axes)).ravel()
        if axes.size != n_dim:
            raise ValueError(f"axes must have length {n_dim}, got {axes.size}")

        label_fontsize = scale_font(plt.rcParams["axes.labelsize"], n_dim)
        tick_labelsize = scale_font(plt.rcParams["xtick.labelsize"], n_dim)

        for j, col in enumerate(columns):
            ax = axes[j]
            periodic_bnds = periodic.get(col) if periodic else None

            def _prep(d, wts, _bnds=periodic_bnds):
                xx, pp = kde_1d(
                    d, bw=kde_bw, weights=wts, fast=kde_fast,
                    n=kde_num_1d, periodic=_bnds,
                )
                ss = _CREDIBLE_INTERVAL_REGISTRY[statistic](xx, pp, credible_interval)
                dw = d
                if _bnds is not None:
                    low, high = _bnds
                    dw = low + (d - low) % (high - low)
                return xx, pp, ss, (dw.min(), dw.max())

            for c_idx, (chain_here, color) in enumerate(zip(_chains, row_colors)):
                y_pos = num_chains - 1 - c_idx  # first chain on top

                if not split:
                    data = chain_here.get(col)
                    if data is None:
                        continue
                    x, pdf, stats, whisker = _prep(data, _weights[c_idx])
                    _draw_violin(
                        ax, x, pdf, y_pos, pdf.max(), violin_width, color, 0,
                        kde_kwargs, stats, whisker, show_extrema,
                    )
                    continue

                data = chain_here.get(col)
                data2 = _chains2[c_idx].get(col)
                top = _prep(data, _weights[c_idx]) if data is not None else None
                bot = _prep(data2, _weights2_norm[c_idx]) if data2 is not None else None
                maxes = [part[1].max() for part in (top, bot) if part is not None]
                if not maxes:
                    continue
                common = max(maxes)
                if top is not None:
                    xt, pt, st, wt = top
                    _draw_violin(
                        ax, xt, pt, y_pos, common, violin_width, color, 1,
                        kde_kwargs, st, wt, show_extrema,
                    )
                if bot is not None:
                    xb, pb, sb, wb = bot
                    _draw_violin(
                        ax, xb, pb, y_pos, common, violin_width, color, -1,
                        {**kde_kwargs, **_split_kwargs}, sb, wb, show_extrema,
                    )

            if truths is not None and truths.get(col) is not None:
                ax.axvline(truths[col], zorder=6, **truth_kwargs)

            ax.set_xlabel(col, fontsize=label_fontsize)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
            ax.xaxis.offsetText.set_fontsize(tick_labelsize)
            ax.xaxis.set_major_locator(
                ticker.MaxNLocator(nbins=n_ticks - 1, prune=None, min_n_ticks=n_ticks)
            )

            ax.set_ylim(-0.7, num_chains - 0.3)
            ax.set_yticks(range(num_chains))
            if j == 0:
                if chain_labels is not None:
                    # tick at y position num_chains - 1 - c belongs to chain c
                    ax.set_yticklabels(chain_labels[::-1])
                else:
                    ax.set_yticklabels([])
            else:
                # hide labels without touching the (possibly shared) formatter
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=tick_labelsize)
            ax.tick_params(axis="y", direction=plt.rcParams["xtick.direction"])

        fig.align_labels()
        fig.tight_layout()

        if mappable is not None:
            fig.colorbar(
                mappable, ax=axes.tolist(), pad=0.02, label=colorbar_label, aspect=30,
            )

        if savepath is not None:
            plt.savefig(savepath)
        return fig, axes


def _normalize_trace_inputs(samples, colors, labels, columns) -> tuple:
    """
    Normalize ``traceplot`` inputs to a list of dicts mapping parameter names to
    ``(nsteps, nwalkers)`` arrays, preserving the walker axis.

    Follows the package shape convention: the last axis of an array is always the
    parameter axis (3D = (nsteps, nwalkers, ndim), 2D = (nsteps, ndim) with a
    single walker), while dict values carry one axis fewer (2D = (nsteps,
    nwalkers), 1D = (nsteps,) with a single walker).
    """
    chain_labels = labels

    if not isinstance(samples, list):
        samples = [samples]
        if isinstance(chain_labels, str):
            chain_labels = [chain_labels]

    num_chains = len(samples)

    _chains = []
    for chain in samples:
        if isinstance(chain, dict):
            tmp = {}
            for k, v in chain.items():
                v = np.asarray(v)
                if v.ndim == 1:
                    v = v.reshape(-1, 1)
                elif v.ndim != 2:
                    raise ValueError(
                        f"Dict values must be (nsteps,) or (nsteps, nwalkers), got shape {v.shape} for {k!r}"
                    )
                tmp[k] = v
        else:
            arr = np.asarray(chain)
            if arr.ndim == 2:
                arr = arr[:, None, :]  # single walker: (nsteps, ndim) -> (nsteps, 1, ndim)
            elif arr.ndim != 3:
                raise ValueError(
                    f"Array chains must be (nsteps, ndim) or (nsteps, nwalkers, ndim), got shape {arr.shape}"
                )
            parameter_labels = [r"$\theta_{" + str(i) + "}$" for i in range(arr.shape[-1])]
            tmp = {k: arr[:, :, i] for i, k in enumerate(parameter_labels)}
        _chains.append(tmp)

    if colors is not None:
        if isinstance(colors, str):
            colors = [colors]
        if len(colors) != num_chains:
            raise ValueError("Number of colors does not match the number of chains")
    else:
        colors = [DEFAULT_COLORLIST[c % len(DEFAULT_COLORLIST)] for c in range(num_chains)]

    if chain_labels is not None and len(chain_labels) != num_chains:
        raise ValueError("Number of labels does not match the number of chains")

    seen = set()
    all_columns = []
    for chain in _chains:
        for k in chain.keys():
            if k not in seen:
                seen.add(k)
                all_columns.append(k)

    if columns is None:
        columns = all_columns

    n_dim = len(columns)

    return _chains, colors, chain_labels, columns, n_dim, num_chains


def traceplot(  # pylint: disable=too-many-branches too-many-statements too-many-arguments too-many-locals
    samples: dict[str, np.ndarray] | np.ndarray | list[dict[str, np.ndarray] | np.ndarray],
    columns: list[str] | None = None,
    truths: Optional[dict | np.ndarray] = None,
    labels: Optional[str | list[str]] = None,
    colors: Optional[list[str]] = None,
    thin: int = 1,
    burn_in: Optional[int | float] = None,
    rolling_mean: bool = False,
    rolling_window: int = 50,
    trace_kwargs: Optional[dict] = None,
    truth_kwargs: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
    fig: Optional[Figure] = None,
    axes: Optional[np.ndarray] = None,
    stylefile: str | None = None,
    savepath: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Walker trace plot: one panel per parameter, each walker drawn as a thin line versus step.

    Useful to inspect MCMC mixing and convergence for walker-based samplers (e.g. emcee, Eryn).

    Parameters
    ----------
    samples : dict[str, np.ndarray], np.ndarray or list[dict[str, np.ndarray] | np.ndarray]
        A single chain or a list of chains (overlaid in different colors). If `np.ndarrays` are provided instead of dictionaries, the parameters will be labelled as :math:`\\theta_i`.

        The last axis of an array is **always** the parameter/dimension axis; the dict counterpart of an array moves that axis into the keys (one key per dimension), so dict values have exactly one axis fewer:

        ============================== ============================ =====================
        Array form                     Dict counterpart (per key)   Meaning
        ============================== ============================ =====================
        ``(nsteps, ndim)``             ``(nsteps,)``                flat chain
        ``(nsteps, nwalkers, ndim)``   ``(nsteps, nwalkers)``       walker-resolved chain
        ============================== ============================ =====================

        Unlike :meth:`cantuccio.core.cornerplot`, the walker axis is preserved here: each walker is drawn as its own line. Flat inputs are treated as a single walker.
    columns : list[str], optional
        List of parameter names to include in the plot. If None, the union of keys across chains will be used.
    truths : dict or np.ndarray, optional
        Dictionary or array mapping parameter names to their true values, drawn as horizontal lines. If a numpy array is provided, it will be matched to the parameters in order of `columns`.
    labels : str or list[str], optional
        Label(s) for the chain(s) to be used in the legend.
    colors : list[str], optional
        List of colors for the chains. If not provided, the default color list is cycled.
    thin : int, default 1
        Plot every `thin`-th step (the x axis keeps the true step numbers).
    burn_in : int or float, optional
        Burn-in length to shade in grey on every panel. An int is interpreted as a number of steps, a float in (0, 1) as a fraction of the chain length.
    rolling_mean : bool, default False
        If True, overlay the rolling mean across walkers (one solid line per chain) as a convergence visual.
    rolling_window : int, default 50
        Window length (in steps) for the rolling mean.
    trace_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.plot`` for the walker lines (e.g. ``lw``, ``alpha``).
    truth_kwargs : dict, optional
        Keyword arguments for the truth lines.
    legend_kwargs : dict, optional
        Keyword arguments for the legend.
    fig : matplotlib.figure.Figure, optional
        Matplotlib Figure object to use for the plot. If None, a new figure will be created.
    axes : np.ndarray, optional
        Array of matplotlib Axes objects of length ``n_dim``. If None, a new set of axes will be created.
    stylefile : str, optional
        Path to a Matplotlib style file to use for the plot. If None, a default style file included with the package will be used.
    savepath : str, optional
        Path to save the figure. If None, the figure will not be saved.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        The figure and the array of axes (shape ``(n_dim,)``).
    """
    if stylefile is None:
        stylefile = get_stylefile()

    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")

    with plt.style.context(stylefile):
        (_chains, colors, chain_labels, columns, n_dim, num_chains
        ) = _normalize_trace_inputs(samples, colors, labels, columns)

        trace_kwargs = dict(trace_kwargs or {})
        truth_kwargs = {"color": "k", "ls": "--", "lw": 1.2, **dict(truth_kwargs or {})}

        if isinstance(truths, np.ndarray):
            truths = dict(zip(columns, truths))

        max_nsteps = max(
            chain[k].shape[0] for chain in _chains for k in chain
        )

        burn = None
        if burn_in is not None:
            if isinstance(burn_in, float) and 0.0 < burn_in < 1.0:
                burn = int(burn_in * max_nsteps)
            else:
                burn = int(burn_in)

        if fig is None or axes is None:
            figsize = (8.0, 1.6 * n_dim)
            fig, axes = plt.subplots(n_dim, 1, figsize=figsize, sharex=True, squeeze=False)
        axes = np.atleast_1d(np.asarray(axes)).ravel()
        if axes.size != n_dim:
            raise ValueError(f"axes must have length {n_dim}, got {axes.size}")

        label_fontsize = scale_font(plt.rcParams["axes.labelsize"], n_dim)
        tick_labelsize = scale_font(plt.rcParams["xtick.labelsize"], n_dim)

        for i, col in enumerate(columns):
            ax = axes[i]

            for chain_here, color in zip(_chains, colors):
                data = chain_here.get(col)
                if data is None:
                    continue
                nsteps, nwalkers = data.shape
                steps = np.arange(nsteps)

                walker_kwargs = {
                    "color": color,
                    "lw": 0.5,
                    "alpha": max(0.05, min(0.8, 5.0 / nwalkers)),
                    **trace_kwargs,
                }
                ax.plot(steps[::thin], data[::thin], **walker_kwargs)

                if rolling_mean:
                    window = min(rolling_window, nsteps)
                    mean = data.mean(axis=1)
                    smoothed = np.convolve(mean, np.ones(window) / window, mode="valid")
                    ax.plot(steps[window - 1:], smoothed, color=color, lw=1.5, zorder=4)

            if burn is not None and burn > 0:
                ax.axvspan(0, burn, color="0.5", alpha=0.15, zorder=0)

            if truths is not None and truths.get(col) is not None:
                ax.axhline(truths[col], zorder=5, **truth_kwargs)

            ax.set_ylabel(col, fontsize=label_fontsize)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
            ax.yaxis.offsetText.set_fontsize(tick_labelsize)
            ax.tick_params(labelsize=tick_labelsize)
            ax.set_xlim(0, max_nsteps - 1)

        axes[-1].set_xlabel("step", fontsize=label_fontsize)

        if chain_labels is not None:
            handles = [
                plt.Line2D([0], [0], color=colors[c], label=chain_labels[c])
                for c in range(num_chains)
            ]
            if truths is not None:
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        color=truth_kwargs.get("color", "k"),
                        ls=truth_kwargs.get("ls", "--"),
                        label=truth_kwargs.get("label", "Truths"),
                    )
                )
            _legend_kwargs = {
                "fontsize": scale_font(plt.rcParams["legend.fontsize"], num_dim=n_dim),
                "frameon": False,
                "fancybox": True,
                "loc": "upper right",
                "ncol": len(handles),
            }
            _legend_kwargs.update(legend_kwargs or {})
            fig.legend(handles=handles, **_legend_kwargs)

        fig.align_labels()
        fig.tight_layout()

        if savepath is not None:
            plt.savefig(savepath)
        return fig, axes
