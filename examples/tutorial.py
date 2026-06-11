import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from cantuccio import cornerplot

    import numpy as np
    import scipy
    import matplotlib.pyplot as plt

    import marimo as mo

    return cornerplot, mo, np, plt, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start generating some data
    """)
    return


@app.cell
def _(np, scipy):
    def get_samples(means, vars, size):
        samples = []
        labels = []

        for i, (mu, var) in enumerate(zip(means, vars)):
            print(f"adding component {i}")
            _samples = scipy.stats.distributions.norm(mu, var).rvs(size)
            samples.append(_samples)
            labels.append(rf"$\mu_{i}$")

        return dict(zip(labels, np.stack(samples, axis=0)))

    return (get_samples,)


@app.cell
def _(get_samples):
    mus = [5, 3, 10]

    first_chain = get_samples(mus, [2, 1, 5], 1000)
    return first_chain, mus


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## let's see how they look
    """)
    return


@app.cell
def _(cornerplot, first_chain, plt):
    _ = cornerplot(samples=first_chain)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### We can now change the style...
    """)
    return


@app.cell
def _(mo):
    from cantuccio.core import OFFDIAG_MODES
    styles = mo.ui.dropdown(options=OFFDIAG_MODES, value='hexbin+kde')
    styles
    return (styles,)


@app.cell
def _(cornerplot, first_chain, plt, styles):
    _ =cornerplot(samples=first_chain, offdiag_mode=styles.value)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ... change the KDE estimation method to use `KDEpy`'s `FFTKDE`...
    """)
    return


@app.cell
def _(cornerplot, first_chain, plt, styles):
    _fig, _axs =cornerplot(samples=first_chain, offdiag_mode=styles.value, kde_kwargs={"fast": False})
    _ =cornerplot(samples=first_chain, offdiag_mode=styles.value, kde_kwargs={"fast": True}, fig=_fig, axes=_axs, colors=["red"])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ... add the true values...
    """)
    return


@app.cell
def _(first_chain, mus):
    truths = dict(zip(first_chain.keys(), mus))
    return (truths,)


@app.cell
def _(cornerplot, first_chain, plt, truths):
    _ = cornerplot(samples=first_chain, truths=truths)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ... plot the difference between data and injection ...
    """)
    return


@app.cell
def _(cornerplot, first_chain, plt, truths):
    _ = cornerplot(samples=first_chain, truths=truths, plot_delta=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ... add a second chain ...
    """)
    return


@app.cell
def _(get_samples, mus):
    second_chain = get_samples(mus, [1.2, 4.1, 3.7], 1000)
    return (second_chain,)


@app.cell
def _(cornerplot, first_chain, plt, second_chain, truths):
    _ = cornerplot(samples=[first_chain, second_chain], truths=truths)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ... and label them ...
    """)
    return


@app.cell
def _(cornerplot, first_chain, plt, second_chain, truths):
    _ = cornerplot(samples=[first_chain, second_chain], truths=truths, labels=['first chain', 'second chain'])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### If the ticks overlap, we can decrease their number
    """)
    return


@app.cell
def _(cornerplot, first_chain, plt, second_chain, truths):
    _ = cornerplot(samples=[first_chain, second_chain], truths=truths, labels=['first chain', 'second chain'], n_ticks=3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Chains do not have to share the same parameters, but we can still plot them together
    """)
    return


@app.cell
def _(cornerplot, first_chain, plt, second_chain, truths):
    _second_chain = second_chain.copy()
    _second_chain[r"$\mu_4$"] = _second_chain[r"$\mu_1$"]

    _second_chain.pop(r"$\mu_1$")

    _truths = truths.copy()
    _truths[r"$\mu_4$"] = truths[r"$\mu_1$"]

    _ = cornerplot(samples=[first_chain, _second_chain], truths=_truths, labels=['first chain', 'second chain'], n_ticks=3)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Now let's see how the corner plot looks for higly correlated parameters under the two KDE estimation methods.
    """)
    return


@app.cell
def _(np):
    cov = np.array([[1, 0.99], [0.99, 1]])
    x, y = np.random.multivariate_normal([0, 0], cov, size=1000).T
    samples = {"x": x, "y": y}
    return (samples,)


@app.cell
def _(cornerplot, plt, samples):
    _fig, _axs = cornerplot(samples=samples, offdiag_mode='kde', labels='scipy KDE', )
    _ = cornerplot(samples=samples, offdiag_mode='kde', kde_kwargs={"fast": True, "alpha": 0.5}, fig=_fig, axes=_axs, colors="red", labels='KDEpy FFTKDE')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### And against the histogram:
    """)
    return


@app.cell
def _(cornerplot, plt, samples):
    _fig, _axs = cornerplot(samples=samples, offdiag_mode='kde', diag_mode='hist', colors="k", labels='histogram')
    _fig, _axs = cornerplot(samples=samples, offdiag_mode='kde', kde_kwargs={"fast": False, "alpha": 0.5}, labels='scipy KDE', fig=_fig, axes=_axs)
    _fig, _axs = cornerplot(samples=samples, offdiag_mode='kde', kde_kwargs={"fast": True, "alpha": 0.5}, fig=_fig, axes=_axs, colors="red", labels='KDEpy FFTKDE')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Using KDEs or histograms can provide slightly different estimated credible intervals.
    """)
    return


@app.cell
def _(cornerplot, plt, samples):
    _fig, _axs = cornerplot(samples=samples, diag_mode='hist', colors="k", labels='histogram')
    _fig, _axs = cornerplot(samples=samples, labels='KDE', fig=_fig, axes=_axs)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Median-symmetric vs highest-density credible intervals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For symmetric, unimodal distributions, the two methods give the same result. However, for skewed or multimodal distributions, they can differ significantly:
    """)
    return


@app.cell
def _(cornerplot, plt, samples):
    _fig, _axs = cornerplot(samples=samples, statistic="median", labels='median-symmetric')
    _ = cornerplot(samples=samples, statistic="hdi", labels='highest-density', fig=_fig, axes=_axs, colors="red")
    plt.show()
    return


@app.cell
def _(get_samples, np):
    # let's generate multimodal samples
    multi_mus = [5, 3, 10]
    multi_vars = [2, 1, 5]
    component_1 = get_samples(multi_mus, multi_vars, 1000)
    multi_mus = [10, 12, 20]
    multi_vars = [0.2, 1, 0.5]
    component_2 = get_samples(multi_mus, multi_vars, 1000)

    multi_samples = {k: np.concatenate([component_1[k], component_2[k]]) for k in component_1.keys()}
    return (multi_samples,)


@app.cell
def _(cornerplot, multi_samples, plt):
    _fig, _axs = cornerplot(samples=multi_samples, statistic="median", labels='median-symmetric')
    _ = cornerplot(samples=multi_samples, statistic="hdi", labels='highest-density', fig=_fig, axes=_axs, colors="red")
    plt.show()
    return


@app.cell
def _(scipy):
    # now let's try with a skewed distribution
    skewed_samples_x = scipy.stats.distributions.skewnorm(a=10).rvs(size=1000)
    skewed_samples_y = scipy.stats.distributions.skewnorm(a=-10).rvs(size=1000)
    skewed_samples = {"x": skewed_samples_x, "y": skewed_samples_y}
    return (skewed_samples,)


@app.cell
def _(cornerplot, plt, skewed_samples):
    _fig, _axs = cornerplot(samples=skewed_samples, statistic="median", labels='median-symmetric')
    _ = cornerplot(samples=skewed_samples, statistic="hdi", labels='highest-density', fig=_fig, axes=_axs, colors="red")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### We can also overlay a covariance matrix ellipse on top of our corner plot if we have it available:
    """)
    return


@app.cell
def _(get_samples, np):
    new_mus = [5, 3, 10]
    new_vars = np.array([2, 1, 5])

    new_chain = get_samples(new_mus, new_vars, 1000)
    return new_chain, new_mus, new_vars


@app.cell
def _():
    from cantuccio.core import overlay_covariance

    return (overlay_covariance,)


@app.cell
def _(cornerplot, new_chain, new_mus, new_vars, np, overlay_covariance, plt):
    _fig, _axs = cornerplot(samples=new_chain, contour_levels=[0.68, 0.90], labels='samples')
    new_cov = np.diag(new_vars**2)
    # Pass credible masses via `levels` (matching `contour_levels`) so the FIM
    # ellipses line up with the corner contours. `num_sigmas` instead draws
    # fixed-radius ellipses: a "1 sigma" ellipse encloses only 39% of the 2D
    # mass, not the 68% that 1 sigma covers in 1D, so it would look too small.
    _fig = overlay_covariance(_fig, new_cov, means=new_mus, plot_1d=True, colors='k', levels=[0.68, 0.90],  label='generating distribution')

    plt.show()
    return


if __name__ == "__main__":
    app.run()
