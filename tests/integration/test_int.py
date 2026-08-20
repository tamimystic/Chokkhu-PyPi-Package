from __future__ import annotations

import numpy as np
import pandas as pd

import chokkhu


def test_integration_pipeline():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 100.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "cat": ["A", "B", "A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0, 1, 0],
        }
    )

    cleaned = chokkhu.clean(df, missing="median", outliers="iqr", duplicates=False)
    processed, state = chokkhu.preprocess(
        cleaned, target="target", scale="standard", encode="onehot"
    )
    transformed = chokkhu.transform(
        processed, target="target", pca=2, resample="smote", smote_k=1
    )
    splits = chokkhu.split(transformed, target="target", test_size=0.2, random_state=42)
    assert len(splits) == 4
