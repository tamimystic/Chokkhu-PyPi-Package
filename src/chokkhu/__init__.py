from .eda import ImageEDA, image
from .eda import tabular as tabular_fn
from .preprocessing.image import ImagePreProcessor


class EDAWrapper:
    @staticmethod
    def image(
        dataset_path: str,
        save_dir: str = None,
    ):
        """
        Runs Exploratory Data Analysis on an image dataset.
        If save_dir is provided, reports will be saved automatically.
        """
        save_reports = True if save_dir else False
        
        # If save_dir is None, we need to pass a default or None to the class
        # Let's pass the default path if we need to, or None if it handles it.
        if save_dir is None:
            save_dir = "chokkhu_outputs/EDA_Reports"
            
        return ImageEDA(
            dataset_path=dataset_path, 
            save_reports=save_reports, 
            save_dir=save_dir
        )

    @staticmethod
    def tabular(
        dataset_path: str,
        target_col: str = None,
        save_dir: str = None,
    ):
        """
        Runs Exploratory Data Analysis on a tabular dataset.
        If save_dir is provided, reports will be saved automatically.
        """
        save_reports = True if save_dir else False
        
        if save_dir is None:
            save_dir = "chokkhu_outputs/EDA_Reports"
            
        return tabular_fn(
            dataset_path=dataset_path,
            target_col=target_col,
            save_reports=save_reports,
            save_dir=save_dir,
        )


class PreprocessingWrapper:
    @staticmethod
    def image(
        datapath: str,
    ):
        """
        Preprocesses an image dataset.
        """
        processor = ImagePreProcessor(datapath=datapath)
        return processor.get_data()

    @staticmethod
    def tabular(dataset_path: str):
        """
        Preprocesses a tabular dataset.
        (Feature coming soon)
        """
        raise NotImplementedError("Tabular preprocessing API is under development.")


eda = EDAWrapper()
preprocessing = PreprocessingWrapper()

__all__ = ["eda", "preprocessing"]
