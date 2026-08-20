from __future__ import annotations

import pandas as pd
import seaborn as sns


class BasePlotter:

    def __init__(
        self,
        df: pd.DataFrame,
        results: dict,
        save_dir: str,
        save_reports: bool,
        target_col: str = None,
    ):
        self.df = df
        self.results = results
        self.save_dir = save_dir
        self.save_reports = save_reports
        self.target_col = target_col
        sns.set_theme(style="whitegrid", palette="muted")

    def _add_bar_labels(self, ax, fmt="%.2f"):
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, padding=3)

    def plot(self):
        raise NotImplementedError("Plot method must be implemented by subclasses.")
