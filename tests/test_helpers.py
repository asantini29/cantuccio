"""Tests for the private helpers behind the plots.

``_normalize_inputs`` is the shared engine and lives in ``cantuccio.core``; the
corner-specific helpers live in ``cantuccio.corner``.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import cantuccio.corner as cornerplot_mod
from cantuccio.core import _normalize_inputs
from cantuccio.corner import (
    _resolve_plot_config,
    _setup_figure,
    _sync_axes,
    _place_legend,
    _plot_diagonal,
    _plot_offdiagonal,
)


# ---------------------------------------------------------------------------
# Task 01-1: _normalize_inputs and _resolve_plot_config
# ---------------------------------------------------------------------------

def test_normalize_inputs_single_chain_coercion():
    np.random.seed(42)
    samples = {"x": np.random.randn(50)}
    _chains, colors, _weights, periodic, chain_labels, columns, truths, n_dim, num_chains = (
        _normalize_inputs(samples, None, None, None, None, None, False, None)
    )
    assert isinstance(_chains, list)
    assert len(_chains) == 1
    assert n_dim == 1
    assert num_chains == 1


def test_normalize_inputs_plot_delta_zeroes_truths():
    np.random.seed(42)
    data = np.random.randn(100)
    data2 = np.random.randn(100)
    samples = {"x": data, "y": data2}
    truths_in = {"x": 1.0, "y": 2.0}
    _chains, colors, _weights, periodic, chain_labels, columns, truths, n_dim, num_chains = (
        _normalize_inputs(samples, None, None, None, None, None, True, truths_in)
    )
    assert all(v == 0.0 for v in truths.values())
    assert all("$" in k for k in truths.keys())


def test_normalize_inputs_mismatched_weights_raises():
    np.random.seed(42)
    chain1 = {"x": np.random.randn(50)}
    chain2 = {"x": np.random.randn(50)}
    samples = [chain1, chain2]
    weights = [np.ones(50)]  # only 1 weight array for 2 chains
    with pytest.raises(ValueError, match="weight"):
        _normalize_inputs(samples, weights, None, None, None, None, False, None)


def test_resolve_plot_config_invalid_diag_mode_raises():
    with pytest.raises(ValueError, match="diag_mode"):
        _resolve_plot_config({}, {}, {}, diag_mode="scatter", offdiag_mode="kde",
                             statistic="hdi", num_chains=1)


def test_resolve_plot_config_invalid_offdiag_mode_raises():
    with pytest.raises(ValueError, match="offdiag_mode"):
        _resolve_plot_config({}, {}, {}, diag_mode="kde", offdiag_mode="bogus",
                             statistic="hdi", num_chains=1)


def test_resolve_plot_config_many_chains_clamps_offdiag_and_warns():
    with pytest.warns(UserWarning):
        result = _resolve_plot_config({}, {}, {}, diag_mode="kde", offdiag_mode="hist",
                                      statistic="hdi", num_chains=6)
    # base_mode is index 3 in the returned tuple
    (kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
     offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d) = result
    assert base_mode == "kde"


def test_resolve_plot_config_does_not_mutate_input_dict():
    kde_kwargs_orig = {"bandwidth": "silverman", "fast": True}
    kde_kwargs_copy = kde_kwargs_orig.copy()
    _resolve_plot_config(kde_kwargs_orig, {}, {}, diag_mode="kde", offdiag_mode="kde",
                         statistic="hdi", num_chains=1)
    assert "bandwidth" in kde_kwargs_orig
    assert kde_kwargs_orig == kde_kwargs_copy


# ---------------------------------------------------------------------------
# Task 01-2: _setup_figure, _sync_axes, _place_legend
# ---------------------------------------------------------------------------

def test_setup_figure_creates_figure_and_hides_upper_triangle():
    fig, axes, all_left_limit, all_right_limit = _setup_figure(
        fig=None, axes=None, n_dim=3, columns=["a", "b", "c"]
    )
    assert fig is not None
    assert axes.shape == (3, 3)
    for i in range(3):
        for j in range(i + 1, 3):
            assert axes[i, j].get_visible() is False
    assert all_left_limit == {"a": float("inf"), "b": float("inf"), "c": float("inf")}
    assert all_right_limit == {"a": float("-inf"), "b": float("-inf"), "c": float("-inf")}
    plt.close("all")


def test_sync_axes_aligns_column_xlims():
    fig, axes, all_left_limit, all_right_limit = _setup_figure(None, None, 2, ["x", "y"])
    axes[0, 0].set_xlim(-3, 3)
    axes[1, 0].set_xlim(-5, 5)
    _sync_axes(axes, n_dim=2, n_ticks=4)
    assert axes[1, 0].get_xlim() == axes[0, 0].get_xlim()
    plt.close("all")


def test_place_legend_adds_legend_when_labels_provided(monkeypatch):
    import matplotlib
    fig, axes, all_left_limit, all_right_limit = _setup_figure(None, None, 2, ["x", "y"])
    # rcParams["legend.fontsize"] may be a non-numeric string (e.g. "medium") in the
    # Agg test session, which causes scale_font to raise TypeError. Set a numeric value.
    monkeypatch.setitem(matplotlib.rcParams, "legend.fontsize", 10)
    _place_legend(fig, chain_labels=["chain A"], colors=["#1f77b4"],
                  truths=None, truth_kwargs={}, num_chains=1, n_dim=2,
                  label_fontsize=10, legend_kwargs=None)
    assert len(fig.legends) == 1
    plt.close("all")


def test_place_legend_no_legend_when_labels_none():
    fig, axes, all_left_limit, all_right_limit = _setup_figure(None, None, 2, ["x", "y"])
    _place_legend(fig, chain_labels=None, colors=["#1f77b4"],
                  truths=None, truth_kwargs={}, num_chains=1, n_dim=2,
                  label_fontsize=10, legend_kwargs=None)
    assert len(fig.legends) == 0
    plt.close("all")


# ---------------------------------------------------------------------------
# Task 01-3: _plot_diagonal and _plot_offdiagonal
# ---------------------------------------------------------------------------

def _build_inputs():
    """Build valid inputs for _plot_diagonal and _plot_offdiagonal via pure helpers."""
    np.random.seed(42)
    samples = {"x": np.random.randn(200), "y": np.random.randn(200)}
    _chains, colors, _weights, periodic, chain_labels, columns, truths, n_dim, num_chains = (
        _normalize_inputs(samples, None, None, None, None, None, False, None)
    )
    (kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
     offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d) = (
        _resolve_plot_config({}, {}, {}, "kde", "kde", "hdi", num_chains)
    )
    fig, axes, all_left_limit, all_right_limit = _setup_figure(None, None, n_dim, columns)
    return (fig, axes, _chains, colors, _weights, chain_labels, columns, n_dim, num_chains,
            kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
            offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d,
            all_left_limit, all_right_limit)


def test_plot_diagonal_runs_without_error():
    (fig, axes, _chains, colors, _weights, chain_labels, columns, n_dim, num_chains,
     kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
     offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d,
     all_left_limit, all_right_limit) = _build_inputs()
    _plot_diagonal(
        axes, _chains, colors, _weights, columns, n_dim, periodic=None,
        diag_mode="kde", credible_interval=0.9, statistic="hdi", base_mode=base_mode,
        kde_bw=kde_bw, kde_fast=kde_fast, kde_num_1d=kde_num_1d, kde_kwargs=kde_kwargs,
        title_format=None, num_chains=num_chains, label_fontsize=10, tick_labelsize=8,
        all_left_limit=all_left_limit, all_right_limit=all_right_limit, xlabelpad=2.0,
        diagonal_ticks=False,
    )
    assert len(axes[0, 0].get_lines()) > 0
    plt.close("all")


def test_plot_diagonal_hist_mode_does_not_call_kde_1d(monkeypatch):
    (fig, axes, _chains, colors, _weights, chain_labels, columns, n_dim, num_chains,
     kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
     offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d,
     all_left_limit, all_right_limit) = _build_inputs()

    def fail(*a, **kw):
        raise AssertionError("kde_1d must not be called in hist mode")

    monkeypatch.setattr(cornerplot_mod, "kde_1d", fail)
    _plot_diagonal(
        axes, _chains, colors, _weights, columns, n_dim, periodic=None,
        diag_mode="hist", credible_interval=0.9, statistic="hdi", base_mode=base_mode,
        kde_bw=kde_bw, kde_fast=kde_fast, kde_num_1d=kde_num_1d, kde_kwargs=kde_kwargs,
        title_format=None, num_chains=num_chains, label_fontsize=10, tick_labelsize=8,
        all_left_limit=all_left_limit, all_right_limit=all_right_limit, xlabelpad=2.0,
        diagonal_ticks=False,
    )
    plt.close("all")


def test_plot_offdiagonal_runs_without_error():
    (fig, axes, _chains, colors, _weights, chain_labels, columns, n_dim, num_chains,
     kde_kwargs, offdiag_kwargs, truth_kwargs, base_mode, overlay_mode,
     offdiag_hist_fn, _use_kde, kde_bw, kde_fast, kde_num_1d, kde_num_2d,
     all_left_limit, all_right_limit) = _build_inputs()
    _plot_offdiagonal(
        axes, _chains, colors, _weights, columns, n_dim, periodic=None,
        offdiag_hist_fn=offdiag_hist_fn, _use_kde=_use_kde, base_mode=base_mode,
        overlay_mode=overlay_mode, kde_bw=kde_bw, kde_fast=kde_fast, kde_num_2d=kde_num_2d,
        contour_levels=[0.68, 0.95], offdiag_kwargs=offdiag_kwargs,
        label_fontsize=10, tick_labelsize=8, xlabelpad=2.0, ylabelpad=2.0,
        diagonal_ticks=False,
    )
    assert len(axes[1, 0].get_lines()) > 0 or len(axes[1, 0].collections) > 0
    plt.close("all")


# ---------------------------------------------------------------------------
# numpy array inputs
# ---------------------------------------------------------------------------

def test_normalize_inputs_ndarray_samples_auto_labels():
    # shape (N, n_params) array — columns should be auto-labelled θ_0, θ_1
    np.random.seed(42)
    data = np.random.randn(100, 2)
    _chains, colors, _weights, periodic, chain_labels, columns, truths, n_dim, num_chains = (
        _normalize_inputs(data, None, None, None, None, None, False, None)
    )
    assert num_chains == 1
    assert n_dim == 2
    # columns labelled as θ_i LaTeX strings
    assert all(r"\theta" in c for c in columns)
    # chain is a dict with those keys
    assert set(_chains[0].keys()) == set(columns)
    assert _chains[0][columns[0]].shape == (100,)


def test_normalize_inputs_ndarray_truths_converted_to_dict():
    np.random.seed(42)
    samples = {"x": np.random.randn(100), "y": np.random.randn(100)}
    truths_arr = np.array([1.0, 2.0])
    _chains, colors, _weights, periodic, chain_labels, columns, truths, n_dim, num_chains = (
        _normalize_inputs(samples, None, None, None, None, ["x", "y"], False, truths_arr)
    )
    assert isinstance(truths, dict)
    assert truths == {"x": 1.0, "y": 2.0}


def test_cornerplot_accepts_ndarray_samples():
    # End-to-end smoke test: ndarray samples reach cornerplot without error
    import matplotlib
    matplotlib.use("Agg")
    from cantuccio import cornerplot
    np.random.seed(42)
    data = np.random.randn(200, 2)
    fig, axes = cornerplot(data)
    assert axes.shape == (2, 2)
    plt.close("all")


def test_cornerplot_accepts_ndarray_truths():
    import matplotlib
    matplotlib.use("Agg")
    from cantuccio import cornerplot
    np.random.seed(42)
    samples = {"x": np.random.randn(200), "y": np.random.randn(200)}
    fig, axes = cornerplot(samples, truths=np.array([0.0, 0.5]))
    assert axes.shape == (2, 2)
    plt.close("all")
