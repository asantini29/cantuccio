"""
corner.py
=========

Corner plot logic.

cornerplot(samples, ...)  ->  (fig, axes)
"""

from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.stats import chi2
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse

from .core import _CREDIBLE_INTERVAL_REGISTRY, _normalize_inputs
from .kde import hdi_levels, kde_1d, kde_2d
from .visuals import (
    chain_cmap,
    get_stylefile,
    reposition_legend,
    scale_font,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

OFFDIAG_MODES = {
    "hist",
    "hexbin",
    "contour",
    "kde",
    "hist+kde",
    "hexbin+kde",
    "contour+kde",
}

DIAG_MODES = {"kde", "hist"}

ALPHA_CREDIBLE_INTERVAL = 0.5

DEFAULT_HIST_CONTOUR_BINS = 20
DEFAULT_HIST_CONTOUR_SMOOTH = 1.0


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


def _hist_1d_fn(data, weights=None, bins=20, range=None):
    return np.histogram(data, bins=bins, range=range, weights=weights, density=True)


def _hist_density_2d(
    x, y, weights=None, bins=DEFAULT_HIST_CONTOUR_BINS, smooth=DEFAULT_HIST_CONTOUR_SMOOTH
):
    """Estimate a 2D density from an optionally smoothed histogram.

    Returns ``(x_out, y_out, z_out)`` on a meshgrid, matching :func:`kde_2d`'s
    orientation (``x_out``/``y_out`` of shape ``(bins, bins)`` from
    :func:`numpy.meshgrid` of the bin centers, ``z_out = H.T``) so the same
    contour-drawing code can consume either estimator. Used to draw contour
    lines in the ``hist``/``hexbin`` off-diagonal modes, where no KDE runs.

    ``smooth`` is the Gaussian sigma in bin units; ``smooth <= 0`` disables
    smoothing. Periodicity is intentionally not handled here: the samples are
    already wrapped into their domain upstream in ``_plot_offdiagonal``, and a
    histogram (unlike a KDE kernel) deposits each sample in exactly one bin, so
    there is no cross-boundary leakage to fold back.
    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
    if smooth and smooth > 0:
        H = gaussian_filter(H, smooth)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    x_out, y_out = np.meshgrid(x_centers, y_centers)
    z_out = H.T
    return x_out, y_out, z_out


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

    if axes.shape != (n_dim, n_dim):
        raise ValueError(
            f"axes.shape {axes.shape!r} does not match the number of columns ({n_dim}x{n_dim})"
        )

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

def cov_ellipse(mean: np.ndarray, cov: np.ndarray, ax: Axes, num_std: int=1.0, **kwargs):
    """
    Plot a covariance ellipse using eigendecomposition. Code from the `https://github.com/mikekatz04/Eryn` package.

    The ellipse axes are aligned with the eigenvectors of the covariance matrix,
    and scaled by sqrt(eigenvalue) * num_std.

    Args:
        mean (array-like): Center of the ellipse (mean_x, mean_y).
        cov (np.ndarray): 2x2 covariance matrix.
        ax (matplotlib.axes.Axes): Axes object on which to plot the ellipse.
        num_std (float, optional): Number of standard deviations for ellipse radius. Default is 1.0.
        **kwargs: Additional keyword arguments passed to matplotlib.patches.Ellipse.

    Returns:
        matplotlib.patches.Ellipse: The covariance ellipse added to the axes.
    """
    # Eigendecomposition: eigenvalues are variances along principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first) for consistent orientation
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Ellipse dimensions: 2 * num_std * sqrt(eigenvalue) for width/height
    width, height = 2 * num_std * np.sqrt(eigenvalues)

    # Rotation angle from the first eigenvector (major axis direction)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    return ax.add_patch(ellipse)

def overlay_covariance(
    fig: Figure,
    covariance: np.ndarray,
    means: Optional[np.ndarray]=None,
    levels: Optional[tuple]=(0.68, 0.90),
    num_sigmas: list=[1, 2, 3],
    plot_1d: bool=False,
    colors: Optional[list | str]="k",
    linestyles: Optional[list | str]=None,
    linewidths: Optional[list | float]=None,
    alpha: float=0.7,
    label: str=None,
):
    """
    Overlay covariance ellipses on corner plot axes. Useful for comparing the results of a Information Matrix analysis to MCMC results.

    For 2D subplots, draws elliptical contours; for 1D diagonal plots, draws vertical lines either side of the mean.

    There are two conventions for sizing the ellipse, controlled by which argument you pass:

    - ``levels`` (recommended for comparing against :meth:`cornerplot`): each entry is a
      **credible mass** (e.g. ``0.68``), matching the ``contour_levels`` argument of
      :meth:`cornerplot`. The ellipse is drawn at the radius that encloses that fraction of
      the 2D probability mass, and the 1D lines at the radius enclosing that fraction of the
      1D mass. Because a 2D region and a 1D interval enclosing the *same* mass sit at
      different radii (:math:`\\chi^2` with 2 vs 1 degrees of freedom), this is the only way to
      make the overlay line up with the corner contours.
    - ``num_sigmas``: each entry is a **Mahalanobis radius** in units of :math:`\\sigma` along
      the principal axes. Note that an ``n``-:math:`\\sigma` *ellipse* encloses only
      :math:`1 - e^{-n^2/2}` of the mass (e.g. ``39%`` at ``1`` :math:`\\sigma`), **not** the
      ``68%`` that ``n=1`` encloses in 1D. Use this only if you specifically want fixed-radius
      ellipses. If ``levels`` is given it takes precedence over ``num_sigmas``.

    Args:
        fig (matplotlib.figure.Figure): Figure object containing corner plot axes.
        covariance (np.ndarray): Covariance matrix from Fisher analysis, shape (n_params, n_params).
        means (np.ndarray, optional): Mean values for each parameter. If None, uses origin (0, 0, ...).
        levels (tuple, optional): Credible masses to plot (e.g., [0.68, 0.90]), matching ``cornerplot``'s ``contour_levels``. Takes precedence over ``num_sigmas``.
        num_sigmas (list, optional): Mahalanobis radii (in sigma) to plot (e.g., [1, 2, 3]). Default [1, 2, 3]. Ignored if ``levels`` is given.
        plot_1d (bool, optional): Whether to plot 1D contours on diagonal plots. Default is False.
        colors (list | str, optional): Colors for each level. Default is "k".
        linestyles (list | str, optional): Line styles for each level. If None, uses solid lines.
        linewidths (list | float, optional): Line widths for each level. If None, uses default (1.5).
        alpha (float, optional): Transparency of contours. Default 0.7.
        label (str, optional): Label for legend entry.

    Returns:
        matplotlib.figure.Figure: The figure with overlaid contours.
    """

    # Convert axs to numpy array if it's a list
    axs = np.array(fig.get_axes())

    # Infer number of parameters from covariance matrix
    n_params = covariance.shape[0]

    if covariance.shape != (n_params, n_params):
        raise ValueError(f"Covariance matrix must be square, got shape {covariance.shape}")

    # The figure must be an (n_params x n_params) corner-plot grid.
    n_axs = int(round(np.sqrt(len(axs))))
    if n_axs != n_params:
        raise ValueError(
            f"covariance has {n_params} parameters but the figure looks like a "
            f"{n_axs}x{n_axs} corner plot ({len(axs)} axes); sizes must match"
        )

    # Set default means to origin
    if means is None:
        means = np.zeros(n_params)
    elif len(means) != n_params:
        raise ValueError(f"means must have length {n_params}, got {len(means)}")

    # Resolve the per-level radii (in units of sigma along the principal axes).
    # A 2D ellipse and a 1D interval enclosing the *same* probability mass sit at
    # different radii (chi-square with df=2 vs df=1), so we keep them separate.
    if levels is not None:
        levels = np.atleast_1d(levels).astype(float)
        radii_2d = np.sqrt(chi2.ppf(levels, df=2))
        radii_1d = np.sqrt(chi2.ppf(levels, df=1))
        n_levels = len(levels)
    else:
        num_sigmas = np.atleast_1d(num_sigmas).astype(float)
        radii_2d = num_sigmas
        radii_1d = num_sigmas
        n_levels = len(num_sigmas)

    # Normalize per-level styling. Each may be given as a single value (broadcast
    # to every level) or as a per-level sequence (validated against n_levels).
    if colors is None:
        colors = [f'C{i}' for i in range(n_levels)]
    elif isinstance(colors, str):
        colors = [colors] * n_levels
    elif len(colors) != n_levels:
        raise ValueError(f"colors must have length {n_levels}, got {len(colors)}")

    if linestyles is None:
        linestyles = ['-'] * n_levels
    elif isinstance(linestyles, str):
        linestyles = [linestyles] * n_levels
    elif len(linestyles) != n_levels:
        raise ValueError(f"linestyles must have length {n_levels}, got {len(linestyles)}")

    if linewidths is None:
        linewidths = [1.5] * n_levels
    elif isinstance(linewidths, (int, float)):
        linewidths = [linewidths] * n_levels
    elif len(linewidths) != n_levels:
        raise ValueError(f"linewidths must have length {n_levels}, got {len(linewidths)}")

    # Extract standard deviations for 1D plots
    sigmas = np.sqrt(np.diag(covariance))

    # Reshape axes into 2D grid if needed
    if axs.ndim == 1:
        # Corner plot axes are typically returned as 1D array
        # Reshape to (n_axs, n_axs) grid (n_axs == n_params, validated above)
        axs_grid = np.empty((n_axs, n_axs), dtype=object)
        idx = 0
        for i in range(n_axs):
            for j in range(n_axs):
                axs_grid[j, i] = axs[idx]
                idx += 1
    else:
        axs_grid = axs

    # Loop over each requested level
    for r_2d, r_1d, color, ls, lw in zip(
        radii_2d, radii_1d, colors, linestyles, linewidths
    ):
        # Loop over all subplots
        for i in range(n_params):
            for j in range(i, n_params):
                ax = axs_grid[i, j]

                if ax is None:
                    continue

                if i == j:
                    if plot_1d:
                        # 1D diagonal plot - draw vertical lines at mean +/- r_1d * sigma
                        mean_val = means[i]
                        sigma_val = sigmas[i]
                        for sign in (-1, 1):
                            ax.axvline(
                                mean_val + sign * r_1d * sigma_val,
                                color=color,
                                linestyle=ls,
                                linewidth=lw,
                                alpha=alpha,
                                zorder=10,
                            )
                else:
                    # 2D off-diagonal plot - draw ellipse at Mahalanobis radius r_2d
                    # Extract 2x2 subcovariance for parameters i (x-axis) and j (y-axis)
                    cov = np.array(
                        (
                            (covariance[i][i], covariance[i][j]),
                            (covariance[j][i], covariance[j][j]),
                        )
                    )
                    mean = np.array((means[i], means[j]))
                    cov_ellipse(
                        mean, cov, ax, num_std=r_2d,
                        edgecolor=color, facecolor='none', linestyle=ls, linewidth=lw,
                        zorder=10, alpha=alpha,
                    )

    # Add legend entry if label is provided
    if label is not None:
        handles = [
            plt.Line2D(
                [0], [0],
                color=colors[0],
                linestyle=linestyles[0],
                linewidth=linewidths[0],
                label=label,
            )
        ]

        # If there's an existing legend, merge its handles so we keep a single,
        # combined figure legend instead of leaving a stale one behind. Inherit
        # its font size so the merged legend matches the cornerplot legend.
        legend_fontsize = None
        if fig.legends:
            old_leg = fig.legends[0]
            existing_handles = old_leg.legend_handles
            existing_texts = old_leg.get_texts()
            existing_labels = [t.get_text() for t in existing_texts]
            if existing_texts:
                legend_fontsize = existing_texts[0].get_fontsize()

            combined_handles = list(existing_handles)
            for h in handles:
                if h.get_label() not in existing_labels:
                    combined_handles.append(h)
            handles = combined_handles
            old_leg.remove()

        if legend_fontsize is None:
            # No existing legend to inherit from. ``overlay_covariance`` runs
            # outside cornerplot's style context, so ``legend.fontsize`` may be a
            # relative string (e.g. "medium"). Resolve it to points before scaling.
            base_fontsize = FontProperties(
                size=plt.rcParams["legend.fontsize"]
            ).get_size_in_points()
            legend_fontsize = scale_font(base_fontsize, num_dim=n_params)
        fig.legend(
            handles=handles,
            loc="upper left",
            fontsize=legend_fontsize,
            markerscale=max(0.6, 1.2 - 0.05 * n_params),
            frameon=False,
            fancybox=True,
            title_fontsize=legend_fontsize,
        )

        # Move the legend into the empty upper-right triangle of the corner plot.
        reposition_legend(fig, n_params)

    return fig

def _place_legend(fig, chain_labels, colors, truths, truth_kwargs, num_chains, n_dim,
                  label_fontsize, legend_kwargs) -> None:
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


def _sync_axes(axes, n_dim, n_ticks) -> None:
    def _make_locator():
        # nbins = n_ticks-1 intervals → exactly n_ticks tick positions; no pruning
        return ticker.MaxNLocator(
            nbins=n_ticks - 1, prune=None, min_n_ticks=n_ticks
        )

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


def _draw_contours(
    ax, x_out, y_out, z_out, color, contour_levels, filled, lines, offdiag_kwargs
) -> None:
    """Draw HDI contour bands and/or lines for a density grid.

    ``filled`` draws the power-law-alpha ``contourf`` bands (the ``contour``
    base mode); ``lines`` draws the ``contour`` outline. No-op when the grid has
    no usable levels below its maximum. Consumes either a ``kde_2d`` grid or a
    ``_hist_density_2d`` grid — the orientation is identical.
    """
    raw_lvls = hdi_levels(z_out, contour_levels)
    lvls = np.unique(raw_lvls)
    lvls = lvls[lvls < z_out.max()].tolist()
    if len(lvls) == 0:
        return

    if filled:
        r, g, b, _ = to_rgba(color)
        # Power-law scaling: squaring the fraction gives 4:1 inner/outer ratio
        band_colors = [
            (r, g, b, ALPHA_CREDIBLE_INTERVAL * ((k + 1) / len(lvls)) ** 2)
            for k in range(len(lvls))
        ]
        ax.contourf(x_out, y_out, z_out, levels=[*lvls, z_out.max()], colors=band_colors)

    if lines:
        ax.contour(
            x_out, y_out, z_out, levels=lvls, **{"colors": [color], **offdiag_kwargs}
        )


def _plot_offdiagonal(axes, _chains, colors, _weights, columns, n_dim, periodic, offdiag_hist_fn,
                      _use_kde, base_mode, overlay_mode, kde_bw, kde_fast, kde_num_2d,
                      contour_levels, offdiag_kwargs, label_fontsize, tick_labelsize, xlabelpad,
                      ylabelpad, diagonal_ticks) -> None:
    for i in range(1, n_dim):
        for j in range(i):
            ax: plt.Axes = axes[i, j]

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
                    _draw_contours(
                        ax, x_out, y_out, z_out, color, contour_levels,
                        filled=(base_mode == "contour"),
                        lines=(overlay_mode == "kde" or base_mode == "kde"),
                        offdiag_kwargs=offdiag_kwargs,
                    )

            if i == n_dim - 1:
                ax.set_xlabel(columns[j], fontsize=label_fontsize)
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
                ax.xaxis.offsetText.set_fontsize(tick_labelsize)
                rotation_kwargs = {"labelrotation": 45, "labelrotation_mode": "xtick"} if diagonal_ticks else {}
                ax.tick_params(axis="x", **rotation_kwargs)
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


def _plot_diagonal(axes, _chains, colors, _weights, columns, n_dim, periodic, diag_mode,
                   credible_interval, statistic, base_mode, kde_bw, kde_fast, kde_num_1d,
                   kde_kwargs, title_format, num_chains, label_fontsize, tick_labelsize,
                   all_left_limit, all_right_limit, xlabelpad, diagonal_ticks) -> None:
    for i in range(n_dim):
        ax = axes[i, i]
        for c_idx, (chain_here, color) in enumerate(zip(_chains, colors)):

            data = chain_here.get(columns[i])
            if data is None:
                continue
            w = _weights[c_idx]

            periodic_bnds = periodic.get(columns[i]) if periodic else None
            if diag_mode == "kde":
                x, pdf = kde_1d(
                    data,
                    bw=kde_bw,
                    weights=w,
                    fast=kde_fast,
                    n=kde_num_1d,
                    periodic=periodic_bnds,
                )
            else:
                hist_data = data
                hist_range = None
                if periodic_bnds is not None:
                    low, high = periodic_bnds
                    hist_data = low + (data - low) % (high - low)
                    hist_range = periodic_bnds
                pdf, bin_edges = _hist_1d_fn(
                    hist_data,
                    weights=w,
                    bins=20,
                    range=hist_range,
                )
                x = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            lo, med, hi = _CREDIBLE_INTERVAL_REGISTRY[statistic](x, pdf, credible_interval)

            if diag_mode == "kde":
                ax.plot(x, pdf, **{"color": color, **kde_kwargs})
            else:
                ax.stairs(pdf, bin_edges, color=color, **kde_kwargs)

            if periodic_bnds is not None:
                if diag_mode == "kde":
                    data_limits = (data.min(), data.max())
                    # If the data is well within the periodic bounds, set the x-limits to the data range for better visualization. Otherwise, set the x-limits to the periodic bounds.
                    current_left = periodic_bnds[0] if data_limits[0] < periodic_bnds[0] * 1.001 else data_limits[0]
                    current_right = periodic_bnds[1] if data_limits[1] > periodic_bnds[1] * 0.999 else data_limits[1]
                else:
                    current_left, current_right = periodic_bnds

                # now check if the ax has already limits set by the previous chains, and if so, take the union of the limits to ensure all chains are visible
                all_left_limit[columns[i]] = min(all_left_limit[columns[i]], current_left)
                all_right_limit[columns[i]] = max(all_right_limit[columns[i]], current_right)

                ax.set_xlim(all_left_limit[columns[i]], all_right_limit[columns[i]])

            if periodic_bnds is not None:
                if diag_mode == "kde":
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
            else:
                mask = (x >= lo) & (x <= hi)

            if base_mode != "kde":
                if diag_mode == "kde":
                    ax.fill_between(x, pdf, where=mask, color=color, alpha=ALPHA_CREDIBLE_INTERVAL)
                else:
                    ax.fill_between(x, pdf, where=mask, step="mid", color=color, alpha=ALPHA_CREDIBLE_INTERVAL)

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

            rotation_kwargs = {"labelrotation": 45, "labelrotation_mode": "xtick"} if diagonal_ticks else {}
            ax.tick_params(axis="x", **rotation_kwargs)

            ax.xaxis.labelpad = xlabelpad

        ax.tick_params(axis="x", labelsize=tick_labelsize)


def _setup_figure(fig, axes, n_dim, columns) -> tuple:
    if fig is None or axes is None:
        figsize = (2.0 * n_dim, 2.0 * n_dim)
        fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize, squeeze=False)

    # Hide upper triangle
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            axes[i, j].set_visible(False)

    all_left_limit = {k: float("inf") for k in columns}
    all_right_limit = {k: float("-inf") for k in columns}

    return fig, axes, all_left_limit, all_right_limit


def _resolve_plot_config(kde_kwargs, offdiag_kwargs, truth_kwargs, diag_mode, offdiag_mode,
                         statistic, num_chains
) -> tuple:
    kde_kwargs = dict(kde_kwargs or {})
    offdiag_kwargs = dict(offdiag_kwargs or {})
    truth_kwargs = dict(truth_kwargs or {})
    truth_kwargs = {"color": "k", "ls": "--", "lw": 1.2, **truth_kwargs}

    kde_bw = kde_kwargs.pop("bandwidth", "silverman")
    kde_fast = kde_kwargs.pop("fast", True)
    kde_num_1d = kde_kwargs.pop("num_1d", 512)
    kde_num_2d = kde_kwargs.pop("num_2d", 80)

    if diag_mode not in DIAG_MODES:
        raise ValueError(
            f"diag_mode {diag_mode!r} not recognised. Use {DIAG_MODES}"
        )

    if offdiag_mode not in OFFDIAG_MODES:
        raise ValueError(
            f"offdiag_mode {offdiag_mode!r} not recognised. " f"Use {OFFDIAG_MODES}"
        )

    if num_chains > 5 and offdiag_mode != "kde":
        offdiag_mode = "kde"
        warnings.warn(
            f"Too many chains ({num_chains} >5), setting offdiag_mode to 'kde'",
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

    return (kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
            offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d)


def cornerplot(  # pylint: disable=too-many-branches, too-many-statements too-many-arguments too-many-positional-arguments too-many-locals
    samples: dict[str, np.ndarray] | np.ndarray | list[dict[str, np.ndarray] | np.ndarray],
    columns: list[str] | None = None,
    weights: np.ndarray | list[np.ndarray] | None = None,
    truths: Optional[dict | np.ndarray] = None,
    plot_delta: bool = False,
    periodic: Optional[dict[str, tuple[float, float]]] = None,
    credible_interval: float = 0.90,
    statistic: str = "median",
    diag_mode: str = "kde",
    offdiag_mode: str = "hexbin+kde",
    labels: Optional[str | list[str]] = None,
    colors: Optional[list[str]] = None,
    contour_levels: tuple[float, ...] = (0.68, 0.90),
    title_format: Optional[str] = None,
    fig: Optional[Figure] = None,
    axes: Optional[np.ndarray] = None,
    kde_kwargs: Optional[dict] = None,
    offdiag_kwargs: Optional[dict] = None,
    truth_kwargs: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
    n_ticks: int = 4,
    diagonal_ticks: bool = False,
    xlabelpad: Optional[float] = 4.0,
    ylabelpad: Optional[float] = 2.0,
    hspace=0.1,
    wspace=0.1,
    stylefile: Optional[str] = None,
    savepath: Optional[str] = None,
) -> tuple[Figure, np.ndarray]:
    """
    Custom corner plot with KDE marginals and 2D contours/hexbin/histograms.

    Parameters
    ----------
    samples : dict[str, np.ndarray], np.ndarray or list[dict[str, np.ndarray] | np.ndarray]
        A single chain or a list of chains, where each dict maps parameter names to samples. If `np.ndarrays` are provided instead of dictionaries, the parameters will be labelled as :math:`\\theta_i`.

        The last axis of an array is **always** the parameter/dimension axis; the dict counterpart of an array moves that axis into the keys (one key per dimension), so dict values have exactly one axis fewer:

        ============================== ============================ =====================
        Array form                     Dict counterpart (per key)   Meaning
        ============================== ============================ =====================
        ``(nsteps, ndim)``             ``(nsteps,)``                flat chain
        ``(nsteps, nwalkers, ndim)``   ``(nsteps, nwalkers)``       walker-resolved chain
        ============================== ============================ =====================

        Walker-resolved inputs are flattened over the walker axis before plotting. If `weights` are given for a walker-resolved chain, they must already be flat with length ``nsteps * nwalkers``.
    columns : list[str], optional
        List of parameter names to include in the plot. If None, all keys from the first chain will be used.
    weights : np.ndarray or list[np.ndarray], optional
        A single array of weights or a list of arrays of weights corresponding to the chains. If None, samples will be treated as unweighted.
    truths : dict or np.ndarray, optional
        Dictionary or array mapping parameter names to their true values. If provided, these will be overplotted as lines on the plot.
        If a numpy array is provided, it will be matched to the parameters in order of `columns`.
    plot_delta : bool, default False
        If True, plot the difference between the samples and the truths (i.e., Δ = samples - truths) instead of the raw samples. Requires `truths` to be provided.
    credible_interval : float, default 0.90
        Credible interval level for shading the 1D KDE plots on the diagonal.
    statistic : str, default "median"
        Statistic to compute for the titles on the diagonal panels. Options are "median" or "hdi". The latter follows the `ChainConsumer` implementation. Refer to the :meth:`get_credible_interval_median` and :meth:`get_credible_interval_hdi` for details.
    diag_mode : str, default "kde"
        Mode for the 1D distributions on the diagonal panels. Options are:
        - "kde": Kernel density estimates
        - "hist": Histograms
    offdiag_mode : str, default "hexbin+kde"
        Mode for the off-diagonal panels. Options are:
        - "hist": 2D histogram
        - "hexbin": Hexagonal binning
        - "contour": Filled contour plot of the KDE
        - "kde": Non-filled contour plot of the KDE
        - "hist+kde": 2D histogram with KDE contours overlaid
        - "hexbin+kde": Hexagonal binning with KDE contours overlaid
        - "contour+kde": Filled contour plot with KDE contours overlaid
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
    kde_kwargs : dict, optional
        Keyword arguments for the KDE estimation. If not provided, defaults will be used (e.g., bandwidth method "silverman" and FFT-based KDE). Supported keys include:
        - "bandwidth": Bandwidth method for KDE ("scott", "silverman", or a scalar factor).
        - "fast": If True, use FFT-based KDE through the `KDEpy` package for faster computation on large datasets. If False, use the standard Gaussian KDE estimator. Default is False.
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
    ticks_format : str, optional
        Format string for the tick labels. If None, the default Matplotlib formatting will be used.
    diagonal_ticks : bool, default False
        If True, the ticks on the x-axis will be tilted by 45 degrees for better readability.
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
        (_chains, colors, _weights, periodic, chain_labels, columns, truths, n_dim, num_chains
        ) = _normalize_inputs(samples, weights, periodic, colors, labels, columns, plot_delta, truths)

        # ── defaults ──────────────────────────────────────────────────────────────
        (kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
         offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d) = _resolve_plot_config(
            kde_kwargs, offdiag_kwargs, truth_kwargs, diag_mode, offdiag_mode, statistic, num_chains
        )

        # ── figure / axes / fontsizes ──────────────────────────────────────────
        fig, axes, all_left_limit, all_right_limit = _setup_figure(fig, axes, n_dim, columns)
        label_fontsize = scale_font(plt.rcParams["axes.labelsize"], n_dim)
        tick_labelsize = scale_font(plt.rcParams["xtick.labelsize"], n_dim)

        # ── diagonal: 1D marginals ────────────────────────────────────────────────
        _plot_diagonal(axes, _chains, colors, _weights, columns, n_dim, periodic, diag_mode,
                       credible_interval, statistic, base_mode, kde_bw, kde_fast, kde_num_1d,
                       kde_kwargs, title_format, num_chains, label_fontsize, tick_labelsize,
                       all_left_limit, all_right_limit, xlabelpad, diagonal_ticks)

        # ── off-diagonal: configurable 2D density ─────────────────────────────────
        _plot_offdiagonal(axes, _chains, colors, _weights, columns, n_dim, periodic, offdiag_hist_fn,
                          _use_kde, base_mode, overlay_mode, kde_bw, kde_fast, kde_num_2d,
                          contour_levels, offdiag_kwargs, label_fontsize, tick_labelsize, xlabelpad,
                          ylabelpad, diagonal_ticks)

        # Now add truths to the diagonal and off-diagonal panels
        if truths is not None:
            overplot_lines(axes, truths, columns=columns, **truth_kwargs)

        # ── sync axis limits and tick count ───────────────────────────────────────
        _sync_axes(axes, n_dim, n_ticks)

        # ── optional legend ────────────────────────────────────────────────────────
        _place_legend(fig, chain_labels, colors, truths, truth_kwargs, num_chains, n_dim,
                      label_fontsize, legend_kwargs)

        fig.align_labels()
        fig.tight_layout()
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        if savepath is not None:
            plt.savefig(savepath)
        return fig, axes
