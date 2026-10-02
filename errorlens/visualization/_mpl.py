"""Optional matplotlib import, shared style and palette."""

from __future__ import annotations

from typing import Any

from errorlens.utils.optional import require

# Validated reference palette (light mode). Categorical slots are used in fixed order.
BLUE = "#2a78d6"     # primary series / "correct" / subgroup metric
ORANGE = "#eb6834"   # second series / "errors"
AQUA = "#1baf7a"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#ffffff"
BASELINE = "#52514e"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

STYLE: dict[str, Any] = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": TEXT,
    "savefig.facecolor": SURFACE,
}


def mpl() -> Any:
    """Import matplotlib or raise a MissingDependencyError with install instructions."""
    return require("matplotlib", package="matplotlib", extra="viz", feature="Plotting")


def sequential_cmap() -> Any:
    colors = require("matplotlib.colors", package="matplotlib", extra="viz", feature="Plotting")
    return colors.LinearSegmentedColormap.from_list("errorlens_blue", SEQUENTIAL)


def new_figure(figsize: tuple[float, float], managed: bool, nrows: int = 1,
               ncols: int = 1, **kwargs: Any) -> tuple[Any, Any]:
    """Create a figure and axes.

    ``managed=True`` uses pyplot (so ``plt.show()`` and notebooks work); ``managed=False``
    creates a standalone Figure with an Agg canvas — no global state, safe in servers and
    report generation.
    """
    mpl()
    if managed:
        import matplotlib.pyplot as plt

        return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, **kwargs)
    return fig, axes
