from __future__ import annotations

import numpy as np
import pandas as pd

import chokkhu


def test_integration_pipeline():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 100.0],
            "cat": ["A", "B", "A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0, 1, 0],
        }
    )

    cleaned = chokkhu.clean(df, missing="median", outliers="iqr", duplicates=False)
    processed, state = chokkhu.preprocess(
        cleaned, target="target", scale="standard", encode="onehot"
    )
    splits = chokkhu.split(processed, target="target", test_size=0.2, random_state=42)
    assert len(splits) == 4
