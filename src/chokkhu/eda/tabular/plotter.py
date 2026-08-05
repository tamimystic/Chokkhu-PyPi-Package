import pandas as pd
from chokkhu.core.logger import Logger

from .plotters.global_plotter import GlobalPlotter
from .plotters.univariate_plotter import UnivariatePlotter
from .plotters.bivariate_plotter import BivariatePlotter
from .plotters.multivariate_plotter import MultivariatePlotter


class TabularPlotter:
    """
    Facade class that orchestrates all the modular sub-plotters.
    """
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

        # Initialize sub-plotters
        self.global_plotter = GlobalPlotter(df, results, save_dir, save_reports, target_col)
        self.univariate_plotter = UnivariatePlotter(df, results, save_dir, save_reports, target_col)
        self.bivariate_plotter = BivariatePlotter(df, results, save_dir, save_reports, target_col)
        self.multivariate_plotter = MultivariatePlotter(df, results, save_dir, save_reports, target_col)

    def plot_all(self):
        Logger.info("Rendering Ultimate Statistical Visualizations...")
        
        self.global_plotter.plot()

        print("\n1. Univariate Analysis")
        self.univariate_plotter.plot()

        print("\n2. Bivariate Analysis")
        self.bivariate_plotter.plot()

        print("\n3. Multivariate Analysis")
        self.multivariate_plotter.plot()
