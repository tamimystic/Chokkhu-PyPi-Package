import os

import matplotlib.pyplot as plt
import seaborn as sns


class PlotVisualizer:
    @staticmethod
    def setup_theme() -> None:
        """Configures the global plotting theme for all EDA tools."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 18,
            }
        )

    @staticmethod
    def save_and_show(fig, filename: str, save_dir: str, save_reports: bool) -> None:
        """
        Saves the figure in 400 DPI (Publication quality) and displays it.
        """
        plt.tight_layout()
        if save_reports:
            os.makedirs(save_dir, exist_ok=True)
            # High quality 400 DPI for thesis/research papers
            fig.savefig(os.path.join(save_dir, filename), dpi=400, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    @staticmethod
    def add_bar_labels(ax, vertical: bool = True) -> None:
        """Adds numerical value labels on top of or next to seaborn/matplotlib bars."""
        for p in ax.patches:
            if vertical:
                val = p.get_height()
                if val > 0:
                    ax.annotate(
                        f"{int(val)}",
                        (p.get_x() + p.get_width() / 2.0, val),
                        ha="center",
                        va="center",
                        xytext=(0, 8),
                        textcoords="offset points",
                    )
            else:
                val = p.get_width()
                if val > 0:
                    ax.annotate(
                        f"{int(val)}",
                        (val, p.get_y() + p.get_height() / 2.0),
                        ha="left",
                        va="center",
                        xytext=(8, 0),
                        textcoords="offset points",
                    )
