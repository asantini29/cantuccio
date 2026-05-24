"""
visuals.py
==========

This module contains some plotting utility functions for :mod:`cantuccio` corner plots.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure

from typing import Optional
import os

pt = 1.0 / 72.27
golden_ratio = (1.0 + 5.0**0.5) / 2.0

journal_sizes = {
    "prd": {"onecol": 246.0 * pt, "twocol": 510.0 * pt},
    "cqg": {"onecol": 374.0 * pt},  # CQG is only one column
}

DEFAULT_COLORLIST = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

CATEGORICAL_COLORLIST = [
    "#1B6BBF",
    "#E8832A",
    "#7B3FA0",
    "#2A9E8F",
    "#C94040",
    "#F0B429",
    "#5AAADE",
    "#A0522D",
]

SEQUENTIAL_COLORLIST = [
    "#033270",  # deep navy
    "#0B5896",  # dark ocean blue
    "#0A85B8",  # medium blue
    "#08ACBE",  # blue-teal
    "#06CFC2",  # teal
    "#04E8C8",  # turquoise
]

SEQUENTIAL_COLORLIST_v2 = [
    "#0B2038",
    "#1D436D",
    "#3E6D99",
    "#0A85B8",
    "#06CFC2", 
    "#04E8C8"
]

DIVERGING_COLORLIST = [
    "#033270",  # deep navy          ← pole
    "#0A85B8",  # medium blue
    "#A8D8EA",  # pale sky            near-centre
    "#A8EDE6",  # pale mint           near-centre
    "#06CFC2",  # teal
    "#04E8C8",  # vivid turquoise    ← pole
]

DIVERGING_COLORLIST_v2 = [
    "#0B2038",  # deep navy          ← pole
    "#0A85B8",  # medium blue
    "#A8D8EA",  # pale sky            near-centre
    "#A8EDE6",  # pale mint           near-centre
    "#06CFC2",  # teal
    "#04E8C8",  # vivid turquoise    ← pole
]


DEFAULT_STYLEFILE = "corner.mplstyle"


def chain_cmap(color: str) -> LinearSegmentedColormap:
    """
    Linear colormap from fully transparent to ``color``; good for hist2d/hexbin.

    Parameters
    ----------
    color : str
        The color to use for the colormap.

    Returns
    -------
    LinearSegmentedColormap
        The colormap.
    """
    r, g, b, _ = to_rgba(color)
    return LinearSegmentedColormap.from_list("_cc", [(r, g, b, 0.0), (r, g, b, 1.0)])


def get_stylefile(filename: Optional[str] = None) -> str:
    """
    Get the absolute path to a matplotlib style file in the 'mplfiles' directory.

    Parameters
    ----------
    filename : str, optional
        The name of the style file. If None, the default style file is used.

    Returns
    -------
    str
        The absolute path to the style file.
    """
    if filename is None:
        filename = DEFAULT_STYLEFILE

    filepath = os.path.join(os.path.dirname(__file__), "mplfiles", filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Style file '{filename}' not found in 'mplfiles' directory. Absolute path searched: {filepath}"
        )
    return filepath


def get_paper_style(
    journal: str = "prd", cols: str = "onecol", aspect: float = golden_ratio
) -> list[str | dict]:
    """
    Get a style list for use with plt.style.context().

    Parameters
    ----------
    journal : str
        Journal name, must be a key in `journal_sizes`.
    cols : str
        Column key, e.g. 'onecol' or 'twocol'.
    aspect : float
        Height = width / aspect. Defaults to the golden ratio.

    Returns
    -------
    list[str | dict]
        Two-element list: [style_file_or_name, {"figure.figsize": (width, height)}].
    """
    # check that journal and cols are valid keys
    if journal not in journal_sizes:
        raise ValueError(
            f"Journal '{journal}' not found. Valid options: {list(journal_sizes.keys())}"
        )
    if cols not in journal_sizes[journal]:
        raise ValueError(
            f"Column option '{cols}' not found for journal '{journal}'. Valid options: {list(journal_sizes[journal].keys())}"
        )

    style = get_stylefile("paper.mplstyle")
    width = journal_sizes[journal][cols]
    return [style, {"figure.figsize": (width, width / aspect)}]


def legend_bbox(num_dim: int) -> tuple[float, float]:
    """Return (x, y) in figure-relative coordinates for the corner-plot legend.

    Places the legend at the top-left of the upper-right triangle.  The
    target row is num_dim // 3 (capped at 2), and x tracks the leftmost
    valid upper-triangle column at that row (col = row + 1), so the anchor
    never lands inside a data panel for any num_dim.

    Parameters
    ----------
    num_dim : int
        Number of corner-plot dimensions.

    Returns
    -------
    tuple of float
        ``(x, y)`` in figure-fraction coordinates (top-left legend corner).
    """
    cell = 1.0 / num_dim
    target_row = min(2, max(0, num_dim // 2))

    # make sure not to exceed the top and right edges of the figure, even for small num_dim
    x_max = 0.75
    y_min = 0.87
    x = min((target_row + 1.3) * cell, x_max)

    y = y_min + 0.02 * num_dim**0.5
    return (x, y)


def reposition_legend(fig: Figure, num_dim: int) -> None:
    """Find the first legend on *fig* and move it into the upper-right triangle.

    Parameters
    ----------
    fig : Figure
        The figure to reposition the legend in.
    num_dim : int
        Number of corner-plot dimensions.
    fontsize : int | None
        If given, override the legend font size.
    """
    if not fig.legends:
        return
    bbox = legend_bbox(num_dim)
    leg = fig.legends[0]
    leg.set_bbox_to_anchor(bbox, transform=fig.transFigure)

    # set the weight of the legend
    for text in leg.get_texts():
        text.set_fontweight("light")


def scale_font(base: float, num_dim: int, ref: int = 2, exp: float = 0.2) -> float:
    """
    Scale ``base`` font size with ``num_dim`` using a power law anchored at ``ref``.

    Parameters
    ----------
    base : float
        The base font size.
    num_dim : int
        Number of corner-plot dimensions.
    ref : int
        The reference number of dimensions.
    exp : float
        The exponent for the power law.

    Returns
    -------
    float
        The scaled font size.
    """
    return max(5.0, base * (num_dim / ref) ** exp)
