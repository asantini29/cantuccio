import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

# Set CANTUCCIO_FIG_DIR to a directory to save every figure produced by the
# tests (one PNG per figure, named after the test) before it is closed, e.g.
#   CANTUCCIO_FIG_DIR=tests/_figures uv run pytest tests/test_plots.py
FIG_DIR = os.environ.get("CANTUCCIO_FIG_DIR")


@pytest.fixture(autouse=True)
def close_figures(request):
    yield
    if FIG_DIR:
        os.makedirs(FIG_DIR, exist_ok=True)
        for n, num in enumerate(plt.get_fignums()):
            fig = plt.figure(num)
            suffix = f"_{n}" if n else ""
            fig.savefig(
                os.path.join(FIG_DIR, f"{request.node.name}{suffix}.png"),
                dpi=150,
                bbox_inches="tight",
            )
    plt.close("all")
