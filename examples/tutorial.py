import marimo

__generated_with = "0.23.5"
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
    cornerplot(samples=first_chain)
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
    cornerplot(samples=first_chain, offdiag_mode=styles.value)
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
def _(cornerplot, first_chain, truths):
    cornerplot(samples=first_chain, truths=truths)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ... plot the difference between data and injection ...
    """)
    return


@app.cell
def _(cornerplot, first_chain, truths):
    cornerplot(samples=first_chain, truths=truths, plot_delta=True)
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
def _(cornerplot, first_chain, second_chain, truths):
    cornerplot(samples=[first_chain, second_chain], truths=truths)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ... and label them ...
    """)
    return


@app.cell
def _(cornerplot, first_chain, second_chain, truths):
    cornerplot(samples=[first_chain, second_chain], truths=truths, chain_labels=['first chain', 'second chain'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### If the ticks overlap, we can decrease their number
    """)
    return


@app.cell
def _(cornerplot, first_chain, second_chain, truths):
    cornerplot(samples=[first_chain, second_chain], truths=truths, chain_labels=['first chain', 'second chain'], n_ticks=3)
    return


if __name__ == "__main__":
    app.run()
