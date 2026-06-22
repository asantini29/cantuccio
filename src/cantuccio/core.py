"""
core.py
=======

Shared, plot-agnostic engine for :mod:`cantuccio`.

This module holds the pieces used by every plot type:

- :func:`_normalize_inputs` — coerce samples/weights/colors/labels/columns into a
  canonical list-of-dicts form (flattening any walker axis).
- the credible-interval estimators (:func:`get_credible_interval`,
  :func:`get_credible_interval_median`, :func:`get_credible_interval_hdi`) and the
  :data:`_CREDIBLE_INTERVAL_REGISTRY` that maps ``statistic`` names to them.

The individual plots live in their own modules (``cornerplot.py``,
``violinplot.py``, ``traceplot.py``) and import from here.
"""

from __future__ import annotations

import numpy as np

from .visuals import DEFAULT_COLORLIST


def get_credible_interval(
    data: np.ndarray, level: float, weights: np.ndarray | None = None
) -> tuple[float, float, float]:
    """
    Return (lower, median, upper) for a highest-density credible interval.
    Standalone function for users who want to compute credible intervals without going through the kde estimation. This is not used internally by the :meth:`cornerplot` method.

    Parameters
    ----------
    data : np.ndarray
        1D array of samples.
    level : float
        Credible interval level, e.g. 0.90 for a 90% credible interval.
    weights : np.ndarray, optional
        1D array of weights corresponding to the samples.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the lower bound, median, and upper bound of the credible interval.
    """
    lo = 100 * (1.0 - level) / 2.0
    percentiles = [lo, 50.0, 100.0 - lo]
    if weights is None:
        return tuple(np.percentile(data, percentiles))

    i = np.argsort(data)
    d = data[i]
    w = weights[i]
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= np.sum(w)
    return tuple(np.interp(np.array(percentiles) / 100.0, cdf, d))


def get_credible_interval_median(
    x: np.ndarray, pdf: np.ndarray, level: float,
) -> tuple[float, float, float]:
    """
    Return (lower, median, upper). The median is the value of x at which the cumulative distribution function (CDF) reaches 0.5, and the lower and upper bounds are the values of x at which the CDF reaches (1 - level) / 2 and 1 - (1 - level) / 2, respectively.

    Parameters
    ----------
    x : np.ndarray
        1D array of samples.
    pdf : np.ndarray
        1D array of probability density values corresponding to the samples.
    level : float
        Credible interval level, e.g. 0.90 for a 90% credible interval.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the lower bound, median, and upper bound of the credible interval.
    """
    if level >= 1.0:
        raise ValueError("Credible interval level must be less than 1.0")
    vals = [0.5 - level / 2, 0.5, 0.5 + level / 2]
    cdf = pdf.cumsum()
    cdf /= cdf.max()  # Normalize to ensure the cumulative distribution goes from 0 to 1
    bounds = np.interp(vals, cdf, x)
    bounds[1] = 0.5 * (bounds[0] + bounds[2])
    return tuple(bounds)

def get_credible_interval_hdi(
        x: np.ndarray, pdf: np.ndarray, level: float
) -> tuple[float, float, float]:
    """
    Return (lower, center, upper) for the highest-density credible interval.

    Routine adapted from the `ChainConsumer` package: https://samreay.github.io/ChainConsumer/, doi:10.21105/joss.00045.

    Parameters
    ----------
    x : np.ndarray
        1D array of samples.
    pdf : np.ndarray
        1D array of probability density values corresponding to the samples.
    level : float
        Credible interval level, e.g. 0.90 for a 90% credible interval.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the lower bound, center, and upper bound of the highest-density interval.
    """
    cdf = pdf.cumsum()
    cdf /= cdf.max()  # Normalize to ensure the cumulative distribution goes from 0 to 1

    x_in = np.concatenate([[x[0]], x])
    cdf_in = np.concatenate([[0.0], cdf])

    eps = 1e-12
    best_width = float("inf")
    best_lower = float(x_in[0])
    best_upper = float(x_in[-1])
    best_start_mass = 0.0
    best_end_mass = 1.0

    for start_idx, start_mass in enumerate(cdf_in[:-1]):
        target = start_mass + level
        if target > 1.0 + eps:
            break

        end_idx = np.searchsorted(cdf_in, target, side="left")

        # Ensure at least one point is in the interval
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        if end_idx >= cdf_in.size:
            break

        # If still slightly under target, move one step right if possible
        if cdf_in[end_idx] - start_mass < level - eps and end_idx + 1 < cdf_in.size:
            end_idx += 1

        lower = float(x_in[start_idx])
        upper = float(x_in[end_idx])
        width = upper - lower
        if width <= eps:
            continue

        if width < best_width - eps:
            best_width = width
            best_lower = lower
            best_upper = upper
            best_start_mass = float(start_mass)
            best_end_mass = float(cdf_in[end_idx])

    interval_mass = best_end_mass - best_start_mass

    if interval_mass <= eps:
        center = 0.5 * (best_lower + best_upper)

    else:
        center_mass = best_start_mass + 0.5 * interval_mass
        center = float(np.interp(center_mass, cdf_in, x_in, left=best_lower, right=best_upper))

    return best_lower, center, best_upper

_CREDIBLE_INTERVAL_REGISTRY = {
    "median": get_credible_interval_median,
    "hdi": get_credible_interval_hdi,
}


def _normalize_inputs(samples, weights, colors, labels, columns, plot_delta, truths
) -> tuple:
    chain_labels = labels  # give a more descriptive name

    if not isinstance(samples, list):
        samples = [samples]
        if isinstance(chain_labels, str):
            chain_labels = [chain_labels]
        if weights is not None and not isinstance(weights, list):
            weights = [weights]

    num_chains = len(samples)

    # Flatten the walker axis where present. By convention the last axis of an
    # array is always the parameter axis, so a 3D array is (nsteps, nwalkers,
    # ndim) and a 2D dict value is (nsteps, nwalkers).
    flattened = []
    for chain in samples:
        if isinstance(chain, dict):
            flattened.append(
                {k: (v.ravel() if getattr(v, "ndim", 1) > 1 else v) for k, v in chain.items()}
            )
        elif getattr(chain, "ndim", 0) == 3:
            flattened.append(chain.reshape(-1, chain.shape[-1]))
        else:
            flattened.append(chain)
    samples = flattened

    if not isinstance(samples[0], dict):
        num_dim = samples[0].shape[1]
        parameter_labels = [r"$\theta_{" + str(i) + "}$" for i in range(num_dim)]
        samples = [
            dict(zip(parameter_labels, samples[c_idx].T)) for c_idx in range(num_chains)
        ]

    if weights is not None and len(weights) != num_chains:
        raise ValueError(
            "Number of weight arrays does not match the number of chains"
        )
    _weights = weights if weights is not None else [None] * num_chains

    if colors is not None:
        if isinstance(colors, str):
            colors = [colors]
        if len(colors) != num_chains:
            raise ValueError("Number of colors does not match the number of chains")
    else:
        colors = DEFAULT_COLORLIST[:num_chains]

    if chain_labels is not None and len(chain_labels) != num_chains:
        raise ValueError("Number of labels does not match the number of chains")

    # We no longer manually unwrap the data here. The user passes the data wrapped or unwrapped,
    # and KDE evaluates and folds correctly onto [low, high] bounds based on `periodic`.

    if plot_delta and truths is None:
        raise ValueError(
            "A dictionary of true values is required if `plot_delta` is True"
        )

    seen = set()
    all_columns = []
    for chain in samples:
        for k in chain.keys():
            if k not in seen:
                seen.add(k)
                all_columns.append(k)

    if columns is None:
        columns = all_columns

    n_dim = len(columns)

    if isinstance(truths, np.ndarray):
        truths = dict(zip(columns, truths))

    _truths = truths.copy() if truths is not None else {}

    _chains = []
    if plot_delta:

        new_columns = []
        for key in columns:
            if key.startswith("$"):
                new_key = r"$\Delta\," + key[1:]
            else:
                new_key = rf"$\Delta\,${key}"

            new_columns.append(new_key)

        for chain in samples:
            tmp_chain = {}
            for old_key, new_key in zip(columns, new_columns):
                if old_key in chain:
                    tmp_chain[new_key] = chain[old_key] - _truths[old_key]
            _chains.append(tmp_chain)
        columns = new_columns
        truths = {key: 0.0 for key in columns}
    else:
        for chain in samples:
            tmp_chain = {}
            for key in columns:
                if key in chain:
                    tmp_chain[key] = chain[key]
            _chains.append(tmp_chain)

    return (_chains, colors, _weights, chain_labels, columns, truths, n_dim, num_chains)
