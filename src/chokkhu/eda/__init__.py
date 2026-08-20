from __future__ import annotations

from .image.engine import ImageEDA
from .tabular.engine import tabular


def image(
    dataset_path: str,
    save_reports: bool = True,
    save_dir: str = "chokkhu_outputs/EDA_Reports",
):
    return ImageEDA(dataset_path, save_reports, save_dir)


__all__ = ["image", "tabular", "ImageEDA"]
