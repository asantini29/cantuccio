"""
Cantuccio - Yet another corner plot package. The sweetest one around.
"""

from importlib.metadata import PackageNotFoundError, version

from . import core, kde, visuals
from .core import get_credible_interval
from .corner import cornerplot, overplot_lines, overlay_covariance, cov_ellipse
from .violin import violinplot
from .trace import traceplot

__copyright__ = "2026, Alessandro Santini"
__author__ = "Alessandro Santini"
__email__ = "alessandro.santini@aei.mpg.de"

try:
    __version__ = version("cantuccio")
except PackageNotFoundError:
    __version__ = "unknown"
