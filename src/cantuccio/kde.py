"""
kde.py
======

Kernel Density Estimation utilities for :mod:`cantuccio`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE


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


def _scipy_kde_1d(
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


def _scipy_kde_2d(
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


def _fft_kde_1d(
    data: np.ndarray,
    bw: str | float,
    n: int = 512,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D FFT-based KDE; returns (x, pdf).

    Parameters
    ----------
    data : np.ndarray
        1D array of data points.
    bw : str or float
        Bandwidth method passed to :class:`KDEpy.FFTKDE`
        ("ISJ", "scott", "silverman", or a scalar scale factor).
    n : int, optional
        Number of points to evaluate the KDE on. Default is 512.

    Returns
    -------
    x : np.ndarray
        1D array of x values.
    pdf : np.ndarray
        1D array of probability density values.
    """
    _scipy_kde = gaussian_kde(data, bw_method=bw, weights=weights)
    absolute_bw = np.sqrt(_scipy_kde.covariance[0, 0])
    kde = FFTKDE(bw=absolute_bw)
    
    span = data.max() - data.min() or 1.0
    pad = 0.15 * span
    x = np.linspace(data.min() - pad, data.max() + pad, n)
    pdf = kde.fit(data, weights=weights).evaluate(x)
    return x, pdf


def _fft_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bw: str | float,
    n: int = 80,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D FFT-based KDE; returns (X, Y, Z) meshgrid.

    Parameters
    ----------
    x : np.ndarray
        1D array of x values.
    y : np.ndarray
        1D array of y values.
    bw : str or float
        Bandwidth method passed to :class:`KDEpy.FFTKDE`
        ("ISJ", "scott", "silverman", or a scalar scale factor).
    n : int, optional
        Number of points to evaluate the KDE on. Default is 80.

    Returns
    -------
    x_out : np.ndarray
        2D array of x values.
    y_out : np.ndarray
        2D array of y values.
    z_out : np.ndarray
        2D array of probability density values.
    """
    _scipy_kde = gaussian_kde(np.vstack([x, y]), bw_method=bw, weights=weights)
    
    #"Whiten" the data with Cholesky decomposition of the kernel covariance.
    # This elegantly untangles correlations so the data becomes an isotropic blob perfectly
    # suited for KDEpy's axis-aligned FFT algorithms.
    cov = _scipy_kde.covariance
    L = np.linalg.cholesky(cov)

    data_orig = np.vstack([x, y])
    data_white = np.linalg.solve(L, data_orig)

    # Fit FFTKDE using an exact isotropic kernel of bandwidth 1.0 in the whitened space
    kde = FFTKDE(bw=1.0)
    kde.fit(data_white.T, weights=weights)

    # Let KDEpy use its optimal FFT grid locally over this whitened sphere
    grid_white, z_white = kde.evaluate(n)
    
    # Scale density back using the Jacobian determinant
    z_out = z_white.reshape(n, n).T / np.linalg.det(L)
    
    # Notice: x_out and y_out will now be skewed grids, perfectly shaped by cov.
    # plt.contour handles these 2D geometry arrays cleanly.
    xi = np.unique(grid_white[:, 0])
    yi = np.unique(grid_white[:, 1])
    # To maintain structural parity in coordinates, we reconstruct the mapped points
    _X, _Y = np.meshgrid(xi, yi)
    _grid_structured = np.vstack([_X.ravel(), _Y.ravel()])
    _grid_mapped = L @ _grid_structured
    
    x_out = _grid_mapped[0].reshape(n, n)
    y_out = _grid_mapped[1].reshape(n, n)

    return x_out, y_out, z_out


def kde_1d(
    data: np.ndarray,
    bw: str | float,
    n: int = 512,
    weights: np.ndarray | None = None,
    fast: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D KDE; returns (x, pdf).

    Parameters
    ----------
    data : np.ndarray
        1D array of data points.
    bw : str or float
        Bandwidth method passed to KDE function.
    n : int, optional
        Number of points to evaluate the KDE on. Default is 512.
    fast : bool, optional
        Use FFT-based KDE (KDEpy.FFTKDE). Default is False.

    Returns
    -------
    x : np.ndarray
        1D array of x values.
    pdf : np.ndarray
        1D array of probability density values.
    """
    if fast:
        return _fft_kde_1d(data, bw, n, weights)
    else:
        return _scipy_kde_1d(data, bw, n, weights)


def kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bw: str | float,
    n: int = 80,
    weights: np.ndarray | None = None,
    fast: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D KDE; returns (X, Y, Z) meshgrid.

    Parameters
    ----------
    x : np.ndarray
        1D array of x values.
    y : np.ndarray
        1D array of y values.
    bw : str or float
        Bandwidth method passed to KDE function.
    n : int, optional
        Number of points to evaluate the KDE on. Default is 80.
    fast : bool, optional
        Use FFT-based KDE (KDEpy.FFTKDE). Default is False.

    Returns
    -------
    x_out : np.ndarray
        2D array of x values.
    y_out : np.ndarray
        2D array of y values.
    z_out : np.ndarray
        2D array of probability density values.
    """
    if fast:
        return _fft_kde_2d(x, y, bw, n, weights)
    else:
        return _scipy_kde_2d(x, y, bw, n, weights)
