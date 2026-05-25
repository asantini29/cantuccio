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


if __name__ == "__main__":
    app.run()
