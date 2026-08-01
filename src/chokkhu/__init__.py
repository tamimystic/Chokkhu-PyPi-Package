from .eda import ImageEDA, image
from .eda import tabular as tabular_fn
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
        target_col: str = None,
        save_reports: bool = True,
        save_dir: str = "chokkhu_outputs/EDA_Reports",
    ):
        """
        Runs the full Exploratory Data Analysis on the tabular dataset.
        """
        return tabular_fn(
            dataset_path=dataset_path,
            target_col=target_col,
            save_reports=save_reports,
            save_dir=save_dir,
        )


eda = EDAWrapper()

__all__ = ["ImageEDA", "ImagePreProcessor", "eda"]
