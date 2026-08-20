from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from chokkhu.core.logger import Logger

from .encoding import (
    BinaryEncoder,
    FrequencyEncoder,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
from .feature_selection import (
    ANOVASelector,
    CorrelationFilterSelector,
    MutualInfoSelector,
    VarianceThresholdSelector,
)
from .scaling import (
    L2Scaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    get_scaler,
)


class PreprocessorState:

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.target_col = None
        self.encoded_cols = []
        self.num_cols = []
        self.cat_cols = []

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col, enc in self.encoders.items():
            if col not in df.columns:
                continue
            if isinstance(enc, OneHotEncoder):
                encoded_df = enc.transform(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            elif isinstance(enc, BinaryEncoder):
                encoded_df = enc.transform(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:
                df[col] = enc.transform(df[col])
        for col, sc in self.scalers.items():
            if col in df.columns:
                df[col] = sc.transform(df[[col]].values).flatten()
        if self.feature_selector is not None:
            df = self.feature_selector.transform(df)
        return df


def preprocess(
    data: pd.DataFrame,
    target: str = None,
    scale: str = None,
    scale_columns: list = None,
    feature_range: tuple = (0, 1),
    encode: str = None,
    encode_columns: list = None,
    onehot_drop: str = "first",
    onehot_max_categories: int = 20,
    ordinal_order: dict = None,
    select_features: str = None,
    select_k: int = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    verbose: bool = True,
    save_report: bool = False,
    report_dir: str = "./chokkhu_reports/",
) -> Tuple[pd.DataFrame, PreprocessorState]:
    df = data.copy()
    state = PreprocessorState()
    state.target_col = target
    target_series = df[target] if target is not None and target in df.columns else None
    if target is not None and target in df.columns:
        features_df = df.drop(columns=[target])
    else:
        features_df = df
    state.num_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    state.cat_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    cols_to_encode = encode_columns if encode_columns else state.cat_cols
    if encode is not None and cols_to_encode:
        for col in cols_to_encode:
            if col not in features_df.columns:
                continue
            if encode == "label":
                enc = LabelEncoder().fit(features_df[col])
                features_df[col] = enc.transform(features_df[col])
                state.encoders[col] = enc
            elif encode == "onehot":
                enc = OneHotEncoder(
                    drop_first=onehot_drop == "first",
                    max_categories=onehot_max_categories,
                ).fit(features_df[col])
                encoded_df = enc.transform(features_df[col], prefix=col)
                features_df = pd.concat(
                    [features_df.drop(columns=[col]), encoded_df], axis=1
                )
                state.encoders[col] = enc
            elif encode == "target" and target_series is not None:
                enc = TargetEncoder().fit(features_df[col], target_series)
                features_df[col] = enc.transform(features_df[col])
                state.encoders[col] = enc
            elif encode == "frequency":
                enc = FrequencyEncoder().fit(features_df[col])
                features_df[col] = enc.transform(features_df[col])
                state.encoders[col] = enc
            elif encode == "binary":
                enc = BinaryEncoder().fit(features_df[col])
                encoded_df = enc.transform(features_df[col], prefix=col)
                features_df = pd.concat(
                    [features_df.drop(columns=[col]), encoded_df], axis=1
                )
                state.encoders[col] = enc
            elif encode == "ordinal":
                enc = OrdinalEncoder(order_dict=ordinal_order).fit(features_df[col])
                features_df[col] = enc.transform(features_df[col])
                state.encoders[col] = enc
    cols_to_scale = scale_columns if scale_columns else state.num_cols
    if scale is not None and cols_to_scale:
        for col in cols_to_scale:
            if col not in features_df.columns:
                continue
            sc = get_scaler(scale, feature_range=feature_range)
            features_df[col] = sc.fit_transform(features_df[[col]].values).flatten()
            state.scalers[col] = sc
    if select_features is not None:
        if select_features == "variance":
            selector = VarianceThresholdSelector(threshold=variance_threshold).fit(
                features_df
            )
        elif select_features == "correlation":
            selector = CorrelationFilterSelector(threshold=correlation_threshold).fit(
                features_df, target=target_series
            )
        elif select_features == "mutual_info" and target_series is not None:
            selector = MutualInfoSelector(k=select_k or 10).fit(
                features_df, target=target_series
            )
        elif select_features == "anova" and target_series is not None:
            selector = ANOVASelector(k=select_k or 10).fit(
                features_df, target=target_series
            )
        else:
            selector = None
        if selector is not None:
            features_df = selector.transform(features_df)
            state.feature_selector = selector
    if target_series is not None:
        features_df[target] = target_series
    if verbose:
        Logger.info(f"Preprocessed features: {data.shape} -> {features_df.shape}")
    return (features_df, state)


__all__ = [
    "preprocess",
    "PreprocessorState",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "L2Scaler",
    "LabelEncoder",
    "OneHotEncoder",
    "TargetEncoder",
    "FrequencyEncoder",
    "BinaryEncoder",
    "OrdinalEncoder",
    "VarianceThresholdSelector",
    "CorrelationFilterSelector",
    "MutualInfoSelector",
    "ANOVASelector",
]
