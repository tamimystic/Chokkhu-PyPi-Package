from .DeepLearningModel import ImageEDA, ImagePreProcessor


def run_eda(
    dataset_path: str,
    save_reports: bool = True,
    save_dir: str = "chokkhu_outputs/EDA_Reports",
):
    """
    Runs the full Exploratory Data Analysis on the image dataset.
    """
    return ImageEDA(
        dataset_path=dataset_path, save_reports=save_reports, save_dir=save_dir
    )


__all__ = ["ImageEDA", "ImagePreProcessor", "run_eda"]
