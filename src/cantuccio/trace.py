"""
trace.py
========

Walker trace plots for MCMC chains.

traceplot(samples, ...)  ->  (fig, axes)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.figure import Figure

from .visuals import DEFAULT_COLORLIST, get_stylefile, scale_font


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

        Unlike :meth:`cantuccio.corner.cornerplot`, the walker axis is preserved here: each walker is drawn as its own line. Flat inputs are treated as a single walker.
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
                    ax.plot(steps[window - 1:], smoothed, color=color, lw=2, zorder=4)

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
