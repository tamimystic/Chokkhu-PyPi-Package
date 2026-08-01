import os

import pandas as pd

from chokkhu.core.exceptions import DataLoadError, InvalidFormatError
from chokkhu.core.logger import Logger
from chokkhu.core.visualizer import PlotVisualizer

from .plotter import TabularPlotter
from .stats import TabularStats


class TabularEDA:
    def __init__(
        self,
        dataset_path: str,
        save_reports: bool = True,
        save_dir: str = "chokkhu_outputs/EDA_Reports_Tabular",
    ):
        self.dataset_path = dataset_path
        self.save_reports = save_reports
        self.save_dir = save_dir
        self.df = pd.DataFrame()
        self.results = {}

        if self.save_reports:
            os.makedirs(self.save_dir, exist_ok=True)

        PlotVisualizer.setup_theme()
        self._perform_eda()

    def _load_data(self):
        try:
            if self.dataset_path.endswith(".csv"):
                self.df = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith((".xls", ".xlsx")):
                self.df = pd.read_excel(self.dataset_path)
            else:
                raise InvalidFormatError(
                    "Unsupported file format. Please provide a CSV or Excel file."
                )
        except Exception as e:
            raise DataLoadError(f"Error loading dataset: {e}")

    def _perform_eda(self):
        Logger.info(f"Executing Modular Tabular EDA for: {self.dataset_path}")
        self._load_data()

        if self.df.empty:
            Logger.error("Could not load data or dataset is empty.")
            return

        Logger.info("Extracting Statistical Metadata...")
        self.results = TabularStats.extract(self.df)

        plotter = TabularPlotter(
            self.df, self.results, self.save_dir, self.save_reports
        )
        plotter.plot_all()

        if self.save_reports:
            Logger.info(
                f"Tabular EDA Complete! All reports saved in 400 DPI at: {self.save_dir}"
            )
