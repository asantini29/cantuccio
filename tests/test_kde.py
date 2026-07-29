import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from cantuccio.kde import kde_1d, kde_2d


def test_kde_1d_unimodal():
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    
    x_slow, pdf_slow = kde_1d(data, bw="scott", fast=False)
    x_fast, pdf_fast = kde_1d(data, bw="scott", fast=True)
    
    # 1D grids are identical by construction
    np.testing.assert_allclose(x_slow, x_fast)
    # The pdfs should be very close
    np.testing.assert_allclose(pdf_slow, pdf_fast, atol=5e-3, rtol=1e-2)


def test_kde_1d_multimodal():
    np.random.seed(42)
    data = np.concatenate([np.random.normal(-3, 1, 500), np.random.normal(3, 1, 500)])
    
    # Use explicit float bandwidth so we aren't oversmoothing multimodal data
    x_slow, pdf_slow = kde_1d(data, bw=0.5, fast=False)
    x_fast, pdf_fast = kde_1d(data, bw=0.5, fast=True)
    
    np.testing.assert_allclose(x_slow, x_fast)
    np.testing.assert_allclose(pdf_slow, pdf_fast, atol=5e-3, rtol=1e-2)


def test_kde_2d_unimodal():
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 2, 1000)
    
    x_slow, y_slow, z_slow = kde_2d(x, y, bw="scott", fast=False)
    x_fast, y_fast, z_fast = kde_2d(x, y, bw="scott", fast=True)
    
    interp_slow = RegularGridInterpolator(
        (y_slow[:, 0], x_slow[0, :]), z_slow, bounds_error=False, fill_value=0.0
    )
    z_slow_on_fast_grid = interp_slow((y_fast, x_fast))
    
    # Check max densities roughly align
    np.testing.assert_allclose(np.max(z_fast), np.max(z_slow), rtol=0.05)
    
    # Check pointwise similarity (only inside regions with significant density)
    mask = z_slow_on_fast_grid > 0.05 * np.max(z_slow_on_fast_grid)
    np.testing.assert_allclose(
        z_fast[mask], 
        z_slow_on_fast_grid[mask], 
        atol=1e-2, 
        rtol=5e-2
    )


def test_kde_2d_multimodal():
    np.random.seed(42)
    # Generate genuinely correlated cross-shaped blobs to test Cholesky whitening
    mean1, cov1 = [-3, -3], [[1, 0.8], [0.8, 1]]
    mean2, cov2 = [3, 3], [[1, 0.8], [0.8, 1]]
    
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 500).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 500).T
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    
    # Use explicit float bandwidth so we aren't oversmoothing multimodal data
    x_slow, y_slow, z_slow = kde_2d(x, y, bw=0.8, fast=False)
    x_fast, y_fast, z_fast = kde_2d(x, y, bw=0.8, fast=True)
    
    interp_slow = RegularGridInterpolator(
        (y_slow[:, 0], x_slow[0, :]), z_slow, bounds_error=False, fill_value=0.0
    )
    z_slow_on_fast_grid = interp_slow((y_fast, x_fast))
    
    np.testing.assert_allclose(np.max(z_fast), np.max(z_slow), rtol=0.05)

    mask = z_slow_on_fast_grid > 0.05 * np.max(z_slow_on_fast_grid)
    np.testing.assert_allclose(
        z_fast[mask],
        z_slow_on_fast_grid[mask],
        atol=1e-2,
        rtol=5e-2
    )


# ---------------------------------------------------------------------------
# Periodic KDE tests (Plan 02-02)
# ---------------------------------------------------------------------------

def test_kde_1d_periodic_x_bounds():
    np.random.seed(42)
    data = np.random.uniform(0, 10, 1000)
    x, pdf = kde_1d(data, bw="scott", n=256, fast=True, periodic=(0.0, 10.0))
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 10.0)


def test_kde_1d_periodic_normalization():
    np.random.seed(42)
    data = np.random.uniform(0, 10, 1000)
    x, pdf = kde_1d(data, bw="scott", n=512, fast=True, periodic=(0.0, 10.0))
    integral = np.trapezoid(pdf, x)
    assert 0.5 < integral < 1.5


def test_kde_1d_periodic_boundary_continuity():
    np.random.seed(42)
    data = np.random.uniform(0, 10, 1000)
    x, pdf = kde_1d(data, bw="scott", n=512, fast=True, periodic=(0.0, 10.0))
    assert abs(pdf[0] - pdf[-1]) < 0.05 * pdf.mean()


def test_kde_2d_periodic_output_shape():
    np.random.seed(42)
    x_data = np.random.uniform(0, 2 * np.pi, 500)
    y_data = np.random.uniform(0, 2 * np.pi, 500)
    X, Y, Z = kde_2d(
        x_data, y_data, bw="scott", n=32, fast=True,
        periodic_x=(0.0, 2 * np.pi), periodic_y=(0.0, 2 * np.pi),
    )
    assert Z.shape == (32, 32)
    assert np.isclose(X[0, 0], 0.0)
    assert np.isclose(X[0, -1], 2 * np.pi)
    assert np.isclose(Y[0, 0], 0.0)
    assert np.isclose(Y[-1, 0], 2 * np.pi)


# ---------------------------------------------------------------------------
# Histogram-based density estimator (Task 1)
# ---------------------------------------------------------------------------

from cantuccio.corner import _hist_density_2d


def test_hist_density_2d_shapes_and_orientation():
    np.random.seed(0)
    x = np.random.randn(2000)
    y = np.random.randn(2000)
    x_out, y_out, z_out = _hist_density_2d(x, y, bins=20, smooth=0)
    assert x_out.shape == y_out.shape == z_out.shape == (20, 20)
    assert np.all(z_out >= 0)
    # Standard-normal samples peak at the origin, i.e. the central bins.
    iy, ix = np.unravel_index(np.argmax(z_out), z_out.shape)
    assert 6 <= ix <= 13
    assert 6 <= iy <= 13


def test_hist_density_2d_weights_shift_mass():
    x = np.array([0.0, 0.0, 10.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    w = np.array([1.0, 1.0, 100.0, 100.0])
    _, _, z_out = _hist_density_2d(x, y, bins=10, smooth=0, weights=w)
    iy, ix = np.unravel_index(np.argmax(z_out), z_out.shape)
    # The heavily weighted corner is at max-x / max-y -> last bins.
    assert ix >= 7
    assert iy >= 7


def test_hist_density_2d_smooth_zero_matches_raw_histogram():
    np.random.seed(2)
    x = np.random.randn(300)
    y = np.random.randn(300)
    H, _, _ = np.histogram2d(x, y, bins=20)
    _, _, z_out = _hist_density_2d(x, y, bins=20, smooth=0)
    assert np.array_equal(z_out, H.T)


def test_hist_density_2d_smoothing_spreads_peak():
    np.random.seed(1)
    x = np.random.randn(500)
    y = np.random.randn(500)
    _, _, z_raw = _hist_density_2d(x, y, bins=20, smooth=0)
    _, _, z_smooth = _hist_density_2d(x, y, bins=20, smooth=1.5)
    assert z_raw.shape == z_smooth.shape
    assert not np.allclose(z_raw, z_smooth)
    # Smoothing a peaky histogram lowers its maximum.
    assert z_smooth.max() < z_raw.max()
