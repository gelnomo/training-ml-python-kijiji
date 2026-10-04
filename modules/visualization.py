"""Chart helpers with one consistent, colorblind-safe style. Every chart is saved as PNG."""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: charts are written to disk, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from modules.config import FIGURES_DIR  # noqa: E402
from modules.utils import slugify  # noqa: E402

# Palette: categorical slots in fixed order + inks + a blue/gray/red diverging map
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])

THOUSANDS = FuncFormatter(lambda x, _: f"{x:,.0f}")


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "legend.frameon": False,
        }
    )


_apply_style()


class Visualization:
    def __init__(self, path: Path = FIGURES_DIR):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, name: str) -> Path:
        file = self.path / f"{slugify(name)}.png"
        fig.savefig(file, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return file

    def bar(
        self,
        data: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        horizontal: bool = False,
        label_fmt: str = "{:,.0f}",
        xerr: Optional[pd.Series] = None,
        file_name: Optional[str] = None,
    ) -> Path:
        """
        Single-series bar chart with value labels at the end of each bar
        :param data: series, index = categories, values = bar lengths (plotted in given order)
        :param horizontal: horizontal bars (better for long category names)
        :param xerr: optional error bars (same index as data)
        """
        height = max(3.5, 0.38 * len(data)) if horizontal else 4.5
        fig, ax = plt.subplots(figsize=(9, height))
        labels = data.index.astype(str)
        if horizontal:
            bars = ax.barh(labels, data.values, color=BLUE, height=0.65, xerr=xerr,
                           error_kw={"ecolor": INK_SECONDARY, "lw": 1, "capsize": 3})
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_formatter(THOUSANDS)
        else:
            bars = ax.bar(labels, data.values, color=BLUE, width=0.7)
            ax.grid(axis="x", visible=False)
            ax.yaxis.set_major_formatter(THOUSANDS)
            if len(data) > 8:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.bar_label(bars, labels=[label_fmt.format(v) for v in data.values],
                     padding=3, fontsize=8, color=INK_SECONDARY)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.margins(x=0.08 if horizontal else 0.02, y=0.02 if horizontal else 0.1)
        return self._save(fig, file_name or f"bar_{title}")

    def distribution(self, values: pd.Series, title: str, xlabel: str) -> Path:
        """Histogram on the raw scale next to the same data on a log1p scale."""
        fig, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=(11, 4))
        ax_raw.hist(values, bins=80, color=BLUE)
        ax_raw.set(title="Raw scale", xlabel=xlabel, ylabel="Number of ads")
        ax_log.hist(np.log1p(values), bins=60, color=BLUE)
        ax_log.set(title="log(1 + x) scale", xlabel=f"log(1 + {xlabel})", ylabel="Number of ads")
        for ax in (ax_raw, ax_log):
            ax.yaxis.set_major_formatter(THOUSANDS)
            ax.grid(axis="x", visible=False)
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
        fig.tight_layout()
        return self._save(fig, f"distribution_{title}")

    def median_line(
        self, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, bins: int = 20
    ) -> Path:
        """Median of y per quantile bin of x, with the interquartile range as a band."""
        binned = pd.qcut(x, q=bins, duplicates="drop")
        grouped = y.groupby(binned, observed=True)
        stats_df = pd.DataFrame(
            {
                "x": x.groupby(binned, observed=True).median(),
                "median": grouped.median(),
                "q1": grouped.quantile(0.25),
                "q3": grouped.quantile(0.75),
            }
        )
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.fill_between(stats_df["x"], stats_df["q1"], stats_df["q3"], color=BLUE, alpha=0.15, lw=0,
                        label="Interquartile range")
        ax.plot(stats_df["x"], stats_df["median"], color=BLUE, lw=2, marker="o", ms=4, label="Median")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.legend(loc="upper left")
        return self._save(fig, f"line_{title}")

    def heatmap_corr(self, corr: pd.DataFrame, title: str) -> Path:
        """Correlation matrix with a diverging map centred on 0 (lower triangle only)."""
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                    annot_kws={"size": 7}, linewidths=1, linecolor="white", square=True,
                    cbar_kws={"shrink": 0.7, "label": "Spearman correlation"}, ax=ax)
        ax.grid(False)
        ax.set_title(title)
        return self._save(fig, f"heatmap_{title}")

    def predicted_vs_actual(self, y_true, y_pred, title: str) -> Path:
        """Scatter on log-log axes with the perfect-prediction diagonal."""
        y_true, y_pred = np.asarray(y_true) + 1, np.clip(np.asarray(y_pred), 0, None) + 1
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(y_true, y_pred, s=8, alpha=0.25, color=BLUE, edgecolors="none", label="Test ads")
        limits = [1, max(y_true.max(), y_pred.max())]
        ax.plot(limits, limits, color=ORANGE, lw=2, label="Perfect prediction")
        ax.set(xscale="log", yscale="log", title=title, xlabel="Actual visits + 1", ylabel="Predicted visits + 1")
        ax.legend(loc="upper left")
        return self._save(fig, f"scatter_{title}")

    def residuals(self, y_true, y_pred, title: str) -> Path:
        """Log residuals against log predictions; a flat cloud around 0 means no systematic bias."""
        log_pred = np.log1p(np.clip(np.asarray(y_pred), 0, None))
        residual = np.log1p(np.asarray(y_true)) - log_pred
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(log_pred, residual, s=8, alpha=0.25, color=BLUE, edgecolors="none")
        ax.axhline(0, color=ORANGE, lw=2)
        ax.set(title=title, xlabel="log(1 + predicted visits)", ylabel="log residual (actual - predicted)")
        return self._save(fig, f"residuals_{title}")
