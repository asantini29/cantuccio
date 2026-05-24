"""
kde.py
======

Kernel Density Estimation utilities for :mod:`cantuccio`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import gaussian_kde


def hdi_levels(z_out: np.ndarray, levels: Sequence[float]) -> list[float]:
    """
    Density thresholds for Highest Density contours.

    For each p in ``levels``, finds z* such that the fraction of total KDE
    mass in the region {z_out >= z*} equals p.  Returns ascending density values
    ready for ``matplotlib.axes.Axes.contour``.

    Parameters
    ----------
    z_out : np.ndarray
        2D array of probability density values.
    levels : Sequence[float]
        Sequence of probability masses for which to compute contour levels.
        Default is (0.68, 0.90).
    Returns
    -------
    lvls : list[float]
        List of contour levels.
    """
    flat = np.sort(z_out.ravel())[::-1]
    cumfrac = np.cumsum(flat) / flat.sum()
    result = []
    for p in levels:
        idx = min(int(np.searchsorted(cumfrac, p)), len(flat) - 1)
        result.append(float(flat[idx]))
    return sorted(result)


def kde_1d(
    data: np.ndarray,
    bw: str | float,
    n: int = 512,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D Gaussian KDE; returns (x, pdf).

    Parameters
    ----------
    data : np.ndarray
        1D array of data points.
    bw : str or float
        Bandwidth method passed to :class:`scipy.stats.gaussian_kde`
        ("scott", "silverman", or a scalar scale factor).
    n : int, optional
        Number of points to evaluate the KDE on. Default is 512.

    Returns
    -------
    x : np.ndarray
        1D array of x values.
    pdf : np.ndarray
        1D array of probability density values.
    """
    kde = gaussian_kde(data, bw_method=bw, weights=weights)
    span = data.max() - data.min() or 1.0
    pad = 0.15 * span
    x = np.linspace(data.min() - pad, data.max() + pad, n)
    return x, kde(x)


def kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bw: str | float,
    n: int = 80,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D Gaussian KDE; returns (X, Y, Z) meshgrid.

    Parameters
    ----------
    x : np.ndarray
        1D array of x values.
    y : np.ndarray
        1D array of y values.
    bw : str or float
        Bandwidth method passed to :class:`scipy.stats.gaussian_kde`
        ("scott", "silverman", or a scalar scale factor).
    n : int, optional
        Number of points to evaluate the KDE on. Default is 80.
    fast : bool, optional
        Use FFT-based KDE (KDExpress). Default is False.

    Returns
    -------
    x_out : np.ndarray
        2D array of x values.
    y_out : np.ndarray
        2D array of y values.
    z_out : np.ndarray
        2D array of probability density values.
    """
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw, weights=weights)
    px = 0.10 * (x.max() - x.min() or 1.0)
    py = 0.10 * (y.max() - y.min() or 1.0)
    xi = np.linspace(x.min() - px, x.max() + px, n)
    yi = np.linspace(y.min() - py, y.max() + py, n)
    x_out, y_out = np.meshgrid(xi, yi)
    z_out = kde(np.vstack([x_out.ravel(), y_out.ravel()])).reshape(x_out.shape)
    return x_out, y_out, z_out
