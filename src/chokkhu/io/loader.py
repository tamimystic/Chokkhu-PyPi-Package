from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger


def _safe_read_json(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_json(path, **kwargs)
    except Exception:
        return pd.read_json(path, orient="records", **kwargs)


def _load_tabular(path: str, format: str = "auto", **kwargs) -> pd.DataFrame:
    ext = (
        os.path.splitext(path)[1].lower() if format == "auto" else f".{format.lower()}"
    )
    loaders = {
        ".csv": pd.read_csv,
        ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
        ".json": _safe_read_json,
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".feather": pd.read_feather,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported tabular format: {ext}")
    return loaders[ext](path, **kwargs)


def _load_images(
    path: str,
    img_size: tuple = None,
    color_mode: str = "rgb",
    flatten: bool = False,
    normalize: bool = False,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
    verbose: bool = True,
) -> dict:
    if not os.path.isdir(path):
        raise ValueError(f"Image dataset path must be a directory: {path}")
    subdirs = sorted(
        [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    )
    class_names = subdirs if subdirs else ["default"]
    images, labels, file_paths = ([], [], [])
    for idx, class_name in enumerate(class_names):
        folder = os.path.join(path, class_name) if subdirs else path
        files = sorted(
            [
                f
                for f in glob.glob(os.path.join(folder, "*"))
                if os.path.splitext(f)[1].lower() in extensions
            ]
        )
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            if color_mode == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif color_mode == "grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_size is not None:
                img = cv2.resize(
                    img, (img_size[1], img_size[0]) if len(img_size) == 2 else img_size
                )
            if normalize:
                img = img.astype(np.float32) / 255.0
            if flatten:
                img = img.flatten()
            images.append(img)
            labels.append(idx if subdirs else 0)
            file_paths.append(f)
    if verbose:
        Logger.info(f"Loaded {len(images)} images across {len(class_names)} classes.")
    return {
        "X": np.array(images),
        "y": np.array(labels),
        "class_names": class_names,
        "file_paths": file_paths,
    }


def load(
    path: str,
    type: str = "tabular",
    format: str = "auto",
    verbose: bool = True,
    **kwargs,
):
    if type == "tabular":
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        df = _load_tabular(path, format=format, **kwargs)
        if verbose:
            Logger.info(
                f"Loaded tabular dataset: shape={df.shape}, cols={len(df.columns)}"
            )
        return df
    elif type == "image":
        return _load_images(path, verbose=verbose, **kwargs)
    else:
        raise ValueError(f"Unsupported data type: {type}. Use 'tabular' or 'image'.")
