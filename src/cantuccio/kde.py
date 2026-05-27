"""
kde.py
======

Kernel Density Estimation utilities for :mod:`cantuccio`.
"""

from __future__ import annotations

from typing import Sequence
import math

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
from KDEpy import FFTKDE
PAD_VALUE = 0.1

def hdi_levels(z_out: np.ndarray, levels: Sequence[float]) -> list[float]:
    """Density thresholds for Highest Density contours."""
    flat = np.sort(z_out.ravel())[::-1]
    cumfrac = np.cumsum(flat) / flat.sum()
    result = []
    for p in levels:
        idx = min(int(np.searchsorted(cumfrac, p)), len(flat) - 1)
        result.append(float(flat[idx]))
    return sorted(result)

def _get_periodic_shift(x: np.ndarray, low: float, high: float) -> tuple[np.ndarray, float]:
    P = high - low
    x_w = low + (x - low) % P
    angles = (x_w - low) / P * 2 * np.pi
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    mean_x = low + (mean_angle / (2 * np.pi) % 1.0) * P
    center = low + P / 2.0
    shift = center - mean_x
    x_s = low + (x_w + shift - low) % P
    return x_s, shift

def _scipy_kde_1d(
    data: np.ndarray,
    bw: str | float,
    n: int = 512,
    weights: np.ndarray | None = None,
    periodic: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1D Gaussian KDE; returns (x, pdf)."""
    if periodic is not None:
        low, high = periodic
        P = high - low
        
        x_s, shift = _get_periodic_shift(data, low, high)
        kde = gaussian_kde(x_s, bw_method=bw, weights=weights)
        
        x = np.linspace(low, high, n)
        pdf = np.zeros_like(x)
        for k in [-2, -1, 0, 1, 2]:
            pdf += kde(x + shift + k * P)
        return x, pdf

    kde = gaussian_kde(data, bw_method=bw, weights=weights)
    span = data.max() - data.min() or 1.0
    pad = PAD_VALUE * span
    x_grid = np.linspace(data.min() - pad, data.max() + pad, n)
    pdf_grid = kde(x_grid)
    x = np.linspace(data.min(), data.max(), n)
    return x, np.interp(x, x_grid, pdf_grid)

def _scipy_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bw: str | float,
    n: int = 80,
    weights: np.ndarray | None = None,
    periodic_x: tuple[float, float] | None = None,
    periodic_y: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Gaussian KDE; returns (X, Y, Z) meshgrid."""
    x_s, shift_x = x, 0.0
    y_s, shift_y = y, 0.0
    
    if periodic_x is not None:
        x_s, shift_x = _get_periodic_shift(x, periodic_x[0], periodic_x[1])
    if periodic_y is not None:
        y_s, shift_y = _get_periodic_shift(y, periodic_y[0], periodic_y[1])
        
    kde = gaussian_kde(np.vstack([x_s, y_s]), bw_method=bw, weights=weights)

    if periodic_x is not None:
        low_x, high_x = periodic_x
        Px = high_x - low_x
        shifts_x = [-2*Px, -Px, 0.0, Px, 2*Px]
        xi = np.linspace(low_x, high_x, n)
    else:
        px = PAD_VALUE * (x.max() - x.min() or 1.0)
        xi = np.linspace(x.min() - px, x.max() + px, n)
        shifts_x = [0.0]

    if periodic_y is not None:
        low_y, high_y = periodic_y
        Py = high_y - low_y
        shifts_y = [-2*Py, -Py, 0.0, Py, 2*Py]
        yi = np.linspace(low_y, high_y, n)
    else:
        py = PAD_VALUE * (y.max() - y.min() or 1.0)
        yi = np.linspace(y.min() - py, y.max() + py, n)
        shifts_y = [0.0]
        
    x_out, y_out = np.meshgrid(xi, yi)
    z_out = np.zeros_like(x_out)
    
    for sx in shifts_x:
        for sy in shifts_y:
            z_partial = kde(np.vstack([(x_out + shift_x + sx).ravel(), (y_out + shift_y + sy).ravel()]))
            z_out += z_partial.reshape(z_out.shape)
            
    return x_out, y_out, z_out

def _fft_kde_1d(
    data: np.ndarray,
    bw: str | float,
    n: int = 512,
    weights: np.ndarray | None = None,
    periodic: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1D FFT-based KDE; returns (x, pdf)."""
    if periodic is not None:
        low, high = periodic
        P = high - low
        
        x_s, shift = _get_periodic_shift(data, low, high)
        _scipy_kde = gaussian_kde(x_s, bw_method=bw, weights=weights)
        absolute_bw = np.sqrt(_scipy_kde.covariance[0, 0])
        
        kde = FFTKDE(bw=absolute_bw)
        grid_w, z_w = kde.fit(x_s, weights=weights).evaluate(4096)
        
        x = np.linspace(low, high, n)
        pdf = np.zeros_like(x)
        for k in [-2, -1, 0, 1, 2]:
            pdf += np.interp(x + shift + k * P, grid_w, z_w, left=0.0, right=0.0)
        return x, pdf

    _scipy_kde = gaussian_kde(data, bw_method=bw, weights=weights)
    absolute_bw = np.sqrt(_scipy_kde.covariance[0, 0])
    kde = FFTKDE(bw=absolute_bw)
    
    min_val, max_val = data.min(), data.max()
    span = max_val - min_val or 1.0
    pad = PAD_VALUE * span
    x_grid = np.linspace(min_val - pad, max_val + pad, n)
    pdf_grid = kde.fit(data, weights=weights).evaluate(x_grid)

    x = np.linspace(min_val, max_val, n)
    pdf = np.interp(x, x_grid, pdf_grid)
    return x, pdf

def _fft_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bw: str | float,
    n: int = 80,
    weights: np.ndarray | None = None,
    periodic_x: tuple[float, float] | None = None,
    periodic_y: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D FFT-based KDE; returns (X, Y, Z) meshgrid."""
    x_s, shift_x = x, 0.0
    y_s, shift_y = y, 0.0
    
    if periodic_x is not None:
        x_s, shift_x = _get_periodic_shift(x, periodic_x[0], periodic_x[1])
    if periodic_y is not None:
        y_s, shift_y = _get_periodic_shift(y, periodic_y[0], periodic_y[1])

    _scipy_kde = gaussian_kde(np.vstack([x_s, y_s]), bw_method=bw, weights=weights)
    cov = _scipy_kde.covariance
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)
    
    data_white = L_inv @ np.vstack([x_s, y_s])

    kde = FFTKDE(bw=1.0)
    kde.fit(data_white.T, weights=weights)

    grid_white, z_white = kde.evaluate(256)
    z_white = z_white / np.linalg.det(L)
    
    xi_w = np.unique(grid_white[:, 0])
    yi_w = np.unique(grid_white[:, 1])
    Z_w = z_white.reshape(len(xi_w), len(yi_w))
    
    interp = RegularGridInterpolator((xi_w, yi_w), Z_w, bounds_error=False, fill_value=0.0)

    if periodic_x is not None:
        low_x, high_x = periodic_x
        Px = high_x - low_x
        shifts_x = [-2*Px, -Px, 0.0, Px, 2*Px]
        xi = np.linspace(low_x, high_x, n)
    else:
        px = PAD_VALUE * (x.max() - x.min() or 1.0)
        xi = np.linspace(x.min() - px, x.max() + px, n)
        shifts_x = [0.0]

    if periodic_y is not None:
        low_y, high_y = periodic_y
        Py = high_y - low_y
        shifts_y = [-2*Py, -Py, 0.0, Py, 2*Py]
        yi = np.linspace(low_y, high_y, n)
    else:
        py = PAD_VALUE * (y.max() - y.min() or 1.0)
        yi = np.linspace(y.min() - py, y.max() + py, n)
        shifts_y = [0.0]
        
    x_out, y_out = np.meshgrid(xi, yi)
    z_out = np.zeros_like(x_out)
    
    for sx in shifts_x:
        for sy in shifts_y:
            qx = x_out + shift_x + sx
            qy = y_out + shift_y + sy
            pts = L_inv @ np.vstack([qx.ravel(), qy.ravel()])
            z_partial = interp(pts.T)
            z_out += z_partial.reshape(z_out.shape)

    return x_out, y_out, z_out

def kde_1d(
    data: np.ndarray,
    bw: str | float,
    n: int = 512,
    weights: np.ndarray | None = None,
    fast: bool = False,
    periodic: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if fast:
        return _fft_kde_1d(data, bw, n, weights, periodic=periodic)
    else:
        return _scipy_kde_1d(data, bw, n, weights, periodic=periodic)

def kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    bw: str | float,
    n: int = 80,
    weights: np.ndarray | None = None,
    fast: bool = False,
    periodic_x: tuple[float, float] | None = None,
    periodic_y: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fast:
        return _fft_kde_2d(x, y, bw, n, weights, periodic_x=periodic_x, periodic_y=periodic_y)
    else:
        return _scipy_kde_2d(x, y, bw, n, weights, periodic_x=periodic_x, periodic_y=periodic_y)
