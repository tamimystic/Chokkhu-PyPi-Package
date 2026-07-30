from .eda.image import ImageEDA
from .eda.tabular import TabularEDA
from .preprocessing.image import ImagePreProcessor


class EDAWrapper:
    @staticmethod
    def image(
        dataset_path: str,
        save_reports: bool = True,
        save_dir: str = "chokkhu_outputs/EDA_Reports",
    ) -> ImageEDA:
        """
        Runs the full Exploratory Data Analysis on the image dataset.
        """
        return ImageEDA(
            dataset_path=dataset_path, save_reports=save_reports, save_dir=save_dir
        )

    @staticmethod
    def tabular(
        dataset_path: str,
        save_reports: bool = True,
        save_dir: str = "chokkhu_outputs/EDA_Reports",
    ) -> TabularEDA:
        """
        Runs the full Exploratory Data Analysis on the tabular dataset.
        """
        return TabularEDA(
            dataset_path=dataset_path, save_reports=save_reports, save_dir=save_dir
        )


eda = EDAWrapper()

__all__ = ["ImageEDA", "TabularEDA", "ImagePreProcessor", "eda"]
