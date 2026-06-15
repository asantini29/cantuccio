import numpy as np
import pytest

from cantuccio import cornerplot, traceplot, violinplot


def _chain(n=100, shift=0.0):
    return {
        "x": np.random.randn(n) + shift,
        "y": np.random.randn(n) + shift,
    }


# ── violinplot ────────────────────────────────────────────────────────────────


def test_violinplot_single_chain_dict():
    fig, axes = violinplot(_chain())
    assert fig is not None
    assert axes.shape == (2,)


def test_violinplot_multiple_chains_with_weights_and_labels():
    chains = [_chain(), _chain(shift=1.0), _chain(shift=-1.0)]
    weights = [np.random.uniform(0.1, 1.0, 100) for _ in chains]
    fig, axes = violinplot(chains, weights=weights, labels=["a", "b", "c"])
    assert fig is not None
    assert axes.shape == (2,)
    # row labels appear on the leftmost axis only, top row = first chain
    ticklabels = [t.get_text() for t in axes[0].get_yticklabels()]
    assert ticklabels == ["c", "b", "a"]


def test_violinplot_array_input():
    fig, axes = violinplot(np.random.randn(100, 3))
    assert axes.shape == (3,)


def test_violinplot_color_by_adds_colorbar():
    chains = [_chain(), _chain(shift=1.0)]
    fig, axes = violinplot(chains, color_by=np.array([10.0, 500.0]), colorbar_label="SNR")
    assert axes.shape == (2,)
    # 2 parameter panels + 1 colorbar axis
    assert len(fig.axes) == 3


def test_violinplot_colors_and_color_by_raises():
    chains = [_chain(), _chain()]
    with pytest.raises(ValueError, match="not both"):
        violinplot(chains, colors=["r", "b"], color_by=np.array([1.0, 2.0]))


def test_violinplot_truths_and_plot_delta():
    chains = [_chain(), _chain(shift=1.0)]
    truths = {"x": 0.0, "y": 0.5}
    fig, axes = violinplot(chains, truths=truths, plot_delta=True)
    assert fig is not None
    assert axes.shape == (2,)


def test_violinplot_missing_parameter_row_is_skipped():
    chain1 = _chain()
    chain2 = {"x": np.random.randn(100)}  # no "y"
    fig, axes = violinplot([chain1, chain2])
    assert axes.shape == (2,)


def test_violinplot_periodic_executes():
    chains = {"phi": np.random.uniform(0, 2 * np.pi, 200)}
    fig, axes = violinplot(chains, periodic={"phi": (0.0, 2 * np.pi)})
    assert axes.shape == (1,)


def test_violinplot_invalid_statistic():
    with pytest.raises(ValueError, match="statistic"):
        violinplot(_chain(), statistic="mode")


def test_violinplot_more_chains_than_default_colors():
    chains = [_chain(shift=float(i)) for i in range(8)]
    fig, axes = violinplot(chains)
    assert axes.shape == (2,)


# ── traceplot ─────────────────────────────────────────────────────────────────


def test_traceplot_3d_array():
    samples = np.random.randn(200, 8, 3)
    fig, axes = traceplot(samples)
    assert fig is not None
    assert axes.shape == (3,)


def test_traceplot_dict_walker_resolved():
    samples = {"x": np.random.randn(200, 8), "y": np.random.randn(200, 8)}
    fig, axes = traceplot(samples)
    assert axes.shape == (2,)


def test_traceplot_flat_inputs_single_walker():
    # 2D array is (nsteps, ndim); 1D dict values are (nsteps,): both single walker
    fig, axes = traceplot(np.random.randn(200, 3))
    assert axes.shape == (3,)

    fig, axes = traceplot({"x": np.random.randn(200)})
    assert axes.shape == (1,)


def test_traceplot_multiple_chains_with_labels():
    c1 = np.random.randn(200, 8, 2)
    c2 = np.random.randn(150, 4, 2) + 1.0
    fig, axes = traceplot([c1, c2], labels=["run 1", "run 2"])
    assert axes.shape == (2,)
    assert len(fig.legends) == 1


def test_traceplot_burn_in_int_and_fraction():
    samples = np.random.randn(200, 4, 2)
    fig, _ = traceplot(samples, burn_in=50)
    assert fig is not None
    fig, _ = traceplot(samples, burn_in=0.25)
    assert fig is not None


def test_traceplot_rolling_mean_thin_truths():
    samples = np.random.randn(200, 4, 2)
    fig, axes = traceplot(
        samples,
        thin=10,
        rolling_mean=True,
        rolling_window=20,
        truths=np.array([0.0, 0.0]),
    )
    assert axes.shape == (2,)


def test_traceplot_invalid_array_shape():
    with pytest.raises(ValueError, match="Array chains"):
        traceplot(np.random.randn(10, 2, 2, 2))


# ── shape conventions across functions ───────────────────────────────────────


def test_cornerplot_3d_array_flattens():
    samples = np.random.randn(100, 8, 3)
    fig, axes = cornerplot(samples)
    assert axes.shape == (3, 3)

    fig_flat, axes_flat = cornerplot(samples.reshape(-1, 3))
    assert axes_flat.shape == axes.shape


def test_cornerplot_dict_walker_values_flatten():
    walker_chain = {"x": np.random.randn(50, 4), "y": np.random.randn(50, 4)}
    flat_chain = {k: v.ravel() for k, v in walker_chain.items()}

    fig, axes = cornerplot(walker_chain)
    assert axes.shape == (2, 2)
    fig_flat, axes_flat = cornerplot(flat_chain)
    assert axes_flat.shape == axes.shape


def test_violinplot_dict_walker_values_flatten():
    walker_chain = {"x": np.random.randn(50, 4), "y": np.random.randn(50, 4)}
    fig, axes = violinplot(walker_chain)
    assert axes.shape == (2,)


# ── violinplot split mode ─────────────────────────────────────────────────────


def test_violinplot_split_two_dicts():
    fig, axes = violinplot(_chain(), samples2=_chain(shift=1.0))
    assert axes.shape == (2,)


def test_violinplot_split_multiple_chains_with_weights():
    a = [_chain(), _chain(shift=1.0), _chain(shift=-1.0)]
    b = [_chain(shift=0.5), _chain(shift=1.5), _chain(shift=-0.5)]
    wa = [np.random.uniform(0.1, 1.0, 100) for _ in a]
    wb = [np.random.uniform(0.1, 1.0, 100) for _ in b]
    fig, axes = violinplot(a, samples2=b, weights=wa, weights2=wb)
    assert axes.shape == (2,)


def test_violinplot_split_row_count_mismatch_raises():
    a = [_chain(), _chain(shift=1.0)]
    b = [_chain()]  # only one row
    with pytest.raises(ValueError, match="same number of rows"):
        violinplot(a, samples2=b)


def test_violinplot_split_color_by_keeps_colorbar():
    a = [_chain(), _chain(shift=1.0)]
    b = [_chain(shift=0.5), _chain(shift=1.5)]
    fig, axes = violinplot(a, samples2=b, color_by=np.array([10.0, 500.0]))
    # 2 parameter panels + 1 colorbar axis
    assert len(fig.axes) == 3


def test_violinplot_split_missing_parameter_one_side():
    a = _chain()
    b = {"x": np.random.randn(100)}  # no "y" in samples2
    fig, axes = violinplot(a, samples2=b)
    assert axes.shape == (2,)


def test_violinplot_split_periodic_executes():
    a = {"phi": np.random.uniform(0, 2 * np.pi, 200)}
    b = {"phi": np.random.uniform(0, 2 * np.pi, 200)}
    fig, axes = violinplot(a, samples2=b, periodic={"phi": (0.0, 2 * np.pi)})
    assert axes.shape == (1,)


def test_violinplot_split_plot_delta():
    a = [_chain(), _chain(shift=1.0)]
    b = [_chain(shift=0.5), _chain(shift=1.5)]
    truths = {"x": 0.0, "y": 0.5}
    fig, axes = violinplot(a, samples2=b, truths=truths, plot_delta=True)
    assert axes.shape == (2,)


def test_violinplot_split_labels_add_legend():
    a = [_chain(), _chain(shift=1.0)]
    b = [_chain(shift=0.5), _chain(shift=1.5)]
    fig, axes = violinplot(a, samples2=b, split_labels=("prior", "posterior"))
    assert len(fig.legends) == 1
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert texts == ["prior", "posterior"]


def test_violinplot_split_labels_wrong_length_raises():
    a = [_chain()]
    b = [_chain(shift=1.0)]
    with pytest.raises(ValueError, match="length 2"):
        violinplot(a, samples2=b, split_labels=("only-one",))


def test_violinplot_split_hatch_executes():
    a = [_chain()]
    b = [_chain(shift=1.0)]
    fig, axes = violinplot(a, samples2=b, split_kwargs={"hatch": "//"})
    assert axes.shape == (2,)
