from __future__ import annotations

import pandas as pd


def remove_duplicates(
    data: pd.DataFrame, subset: list = None, keep: str = "first"
) -> pd.DataFrame:
    return data.drop_duplicates(subset=subset, keep=keep)
