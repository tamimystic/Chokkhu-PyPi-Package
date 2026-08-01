import pandas as pd

from chokkhu.core.logger import Logger
from chokkhu.eda.tabular.advanced import AdvancedAnalyzer
from chokkhu.eda.tabular.categorical import CategoricalAnalyzer
from chokkhu.eda.tabular.metadata import MetadataAnalyzer
from chokkhu.eda.tabular.missing_data import MissingDataAnalyzer
from chokkhu.eda.tabular.multivariate import MultivariateAnalyzer
from chokkhu.eda.tabular.numerical import NumericalAnalyzer
from chokkhu.eda.tabular.plotter import TabularPlotter
from chokkhu.eda.tabular.specialized import SpecializedAnalyzer


def tabular(
    dataset_path: str,
    target_col: str = None,
    save_reports: bool = False,
    save_dir: str = "chokkhu_reports",
):
    """
    Ultimate Tabular EDA Pipeline
    """
    Logger.info(f"Executing Ultimate Tabular EDA for: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        Logger.error(f"Failed to read dataset: {str(e)}")
        return None

    results = {}

    Logger.info("Extracting Topic 1: Metadata & Structural EDA...")
    results["metadata"] = MetadataAnalyzer.analyze(df)

    Logger.info("Extracting Topic 2: Missing Data & Imputation Impact...")
    results["missing_data"] = MissingDataAnalyzer.analyze(df)

    Logger.info("Extracting Topic 3: Quantitative/Numerical Data EDA...")
    results["numerical"] = NumericalAnalyzer.analyze(df)

    Logger.info("Extracting Topic 4: Qualitative/Categorical Data EDA...")
    results["categorical"] = CategoricalAnalyzer.analyze(df)

    Logger.info("Extracting Topic 5: Bivariate & Multivariate EDA...")
    results["multivariate"] = MultivariateAnalyzer.analyze(df)

    Logger.info("Extracting Topic 6: Specialized Columns EDA...")
    results["specialized"] = SpecializedAnalyzer.analyze(df)

    Logger.info("Extracting Topic 7: Advanced Machine Learning & Target EDA...")
    results["advanced"] = AdvancedAnalyzer.analyze(df, target_col=target_col)

    plotter = TabularPlotter(df, results, save_dir, save_reports, target_col)
    plotter.plot_all()

    Logger.info(f"Tabular EDA Complete! All reports saved in 400 DPI at: {save_dir}")
    return results
