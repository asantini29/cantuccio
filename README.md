# `cantuccio`: yet another corner plot library

<p>
  <img src="logos/cantuccio.png" alt="Cantuccio Logo" width="100" align="left" style="margin-right: 15px;" />
  <a href="docs/index.html">
    <img src="https://img.shields.io/badge/Documentation-Online-blue.svg" alt="Documentation Status">
  </a>
  <br/>
  <a href="https://travis-ci.org/cantuccio/cantuccio">
    <img src="https://travis-ci.org/cantuccio/cantuccio.svg?branch=main" alt="Build Status">
  </a>
  <br/>
  <a href="https://badge.fury.io/py/cantuccio">
    <img src="https://badge.fury.io/py/cantuccio.svg" alt="PyPI version">
  </a>
    <br/>
<br clear="left"/>
</p>



> **can·tuc·cio** | /kan'tutːʃo/ 
> *noun* [masculine]
> 1. an Italian biscuit from Tuscany.
> 2. corner, nook.

`cantuccio` is a Python library for creating corner plots, automatically handling the layout and size of labels, ticks, and legends. It is designed to be flexible and easy to use, allowing users to create publication-quality corner plots with minimal code. Visual choices are inspired by many of the existing corner plot libraries, with particular credit to [`corner`](https://corner.readthedocs.io/en/latest/), [`chainconsumer`](https://samreay.github.io/ChainConsumer/), [`cornetto`](https://cornetto.readthedocs.io/en/latest/), and [`makecorner`](https://github.com/tcallister/makecorner). 

## Installation
You can install `cantuccio` using pip:

This project is managed by [uv](https://docs.astral.sh/uv/). To set up the development environment, install [uv](https://docs.astral.sh/uv/) if needed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, clone the repository and install the package in editable mode:
```bash
git clone https://github.com/asantini29/cantuccio
cd cantuccio
uv sync
uv run pip install -e .
```
Alternatively, you can install the latest version directly from PyPI:
```bash
pip install cantuccio
```

## Example
Here is a simple example of how to use `cantuccio` to create a corner plot:
```python
import numpy as np
import cantuccio as ct
# Generate some random data
data = np.random.randn(1000, 3)
labels = ['x', 'y', 'z']
data_dict = {label: data[:, i] for i, label in enumerate(labels)}
# Create a corner plot
fig, axs = ct.cornerplot(data_dict)
fig.show()
```

## AI Statement
This project made use of generative AI. All the outputs have been reviewed and edited to ensure quality and consistency.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
