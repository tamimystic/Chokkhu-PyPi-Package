from .eda import ImageEDA, image
from .eda import tabular as tabular_fn


class EDAWrapper:
    @staticmethod
    def image(
        dataset_path: str,
        save_reports: bool = False,
        save_dir: str = None,
    ):
        """
        Runs Exploratory Data Analysis on an image dataset.
        """
        if save_reports and save_dir is None:
            save_dir = "chokkhu_outputs/image_reports"

        return ImageEDA(
            dataset_path=dataset_path, save_reports=save_reports, save_dir=save_dir
        )

    @staticmethod
    def tabular(
        dataset_path: str,
        save_reports: bool = False,
        save_dir: str = None,
        target_col: str = None,
    ):
        """
        Runs Exploratory Data Analysis on a tabular dataset.
        """
        if save_reports and save_dir is None:
            save_dir = "chokkhu_outputs/tabular_reports"

        return tabular_fn(
            dataset_path=dataset_path,
            target_col=target_col,
            save_reports=save_reports,
            save_dir=save_dir,
        )


eda = EDAWrapper()

__all__ = ["eda"]
