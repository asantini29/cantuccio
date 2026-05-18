"""
Cantuccio - Yet another corner plot package. The sweetest one around.
"""

from . import core, kde, visuals
from .core import cornerplot, overplot_lines, get_credible_interval

from importlib.metadata import version, PackageNotFoundError

__copyright__ = "2026, Alessandro Santini"
__author__ = "Alessandro Santini"
__email__ = "alessandro.santini@aei.mpg.de"

try:
    __version__ = version("cantuccio")
except PackageNotFoundError:
    __version__ = "unknown"
