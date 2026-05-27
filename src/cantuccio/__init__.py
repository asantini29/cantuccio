"""
Cantuccio - Yet another corner plot package. The sweetest one around.
"""

from importlib.metadata import PackageNotFoundError, version

from . import core, kde, visuals
from .core import cornerplot, overplot_lines

__copyright__ = "2026, Alessandro Santini"
__author__ = "Alessandro Santini"
__email__ = "alessandro.santini@aei.mpg.de"

try:
    __version__ = version("cantuccio")
except PackageNotFoundError:
    __version__ = "unknown"
