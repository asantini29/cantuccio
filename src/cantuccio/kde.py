"""
kde.py
======

Kernel Density Estimation utilities for :mod:`cantuccio`.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from KDExpress.multivariate import fft_kde2d as _fft_kde2d
from KDExpress.univariate import fft_kde1d as _fft_kde1d
from scipy.stats import gaussian_kde

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def _kde1d(
    data: np.ndarray, bw, n: int = 512, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """1D Gaussian KDE; returns (x, pdf)."""
    kde = gaussian_kde(data, bw_method=bw, weights=weights)
    span = data.max() - data.min() or 1.0
    pad = 0.15 * span
    x = np.linspace(data.min() - pad, data.max() + pad, n)
    return x, kde(x)


def _kde2d(
    x: np.ndarray, y: np.ndarray, bw, n: int = 80, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Gaussian KDE; returns (X, Y, Z) meshgrid."""
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw, weights=weights)
    px = 0.10 * (x.max() - x.min() or 1.0)
    py = 0.10 * (y.max() - y.min() or 1.0)
    xi = np.linspace(x.min() - px, x.max() + px, n)
    yi = np.linspace(y.min() - py, y.max() + py, n)
    x_out, y_out = np.meshgrid(xi, yi)
    z_out = kde(np.vstack([x_out.ravel(), y_out.ravel()])).reshape(x_out.shape)
    return x_out, y_out, z_out


def _kde1d_fast(
    data: np.ndarray, bw, n: int = 512, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """1D KDE via FFT convolution (KDExpress); returns (x, pdf)."""
    span = data.max() - data.min() or 1.0
    pad = 0.15 * span
    x = jnp.linspace(float(data.min() - pad), float(data.max() + pad), n)
    _bw = None if isinstance(bw, str) else bw
    _w = jnp.asarray(weights) if weights is not None else None
    pdf = _fft_kde1d(x, jnp.asarray(data), weights=_w, bw=_bw)
    return np.array(x), np.array(pdf)


def _kde2d_fast(
    x: np.ndarray, y: np.ndarray, bw, n: int = 80, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D KDE via FFT convolution (KDExpress); returns (X, Y, Z) meshgrid."""
    px = 0.10 * (x.max() - x.min() or 1.0)
    py = 0.10 * (y.max() - y.min() or 1.0)
    xi = jnp.linspace(float(x.min() - px), float(x.max() + px), n)
    yi = jnp.linspace(float(y.min() - py), float(y.max() + py), n)
    data_2d = jnp.stack([jnp.asarray(x), jnp.asarray(y)], axis=-1)
    _bw = None if isinstance(bw, str) else bw
    _w = jnp.asarray(weights) if weights is not None else jnp.ones(data_2d.shape[0])
    z_out = _fft_kde2d(xi, yi, data_2d, weights=_w, bw=_bw)  # shape (n_x, n_y), ij-indexed
    x_out, y_out = np.meshgrid(np.array(xi), np.array(yi))
    return x_out, y_out, np.array(z_out).T  # .T: ij → xy for matplotlib


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
    fast: bool = False,
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
    fast : bool, optional
        Use FFT-based KDE (KDExpress). Default is False.

    Returns
    -------
    x : np.ndarray
        1D array of x values.
    pdf : np.ndarray
        1D array of probability density values.
    """
    if fast:
        return _kde1d_fast(data, bw, n, weights=weights)
    return _kde1d(data, bw, n, weights=weights)


def kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bw: str | float,
    n: int = 80,
    fast: bool = False,
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
    if fast:
        return _kde2d_fast(x, y, bw, n, weights=weights)
    return _kde2d(x, y, bw, n, weights=weights)
