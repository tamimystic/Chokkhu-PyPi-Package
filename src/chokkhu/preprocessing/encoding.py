import numpy as np
import pandas as pd

class LabelEncoder:
    def __init__(self):
        self.mapping = {}
        self.inverse_mapping = {}

    def fit(self, s: pd.Series):
        uniques = sorted(s.dropna().unique().tolist())
        self.mapping = {val: idx for idx, val in enumerate(uniques)}
        self.inverse_mapping = {idx: val for val, idx in self.mapping.items()}
        return self

    def transform(self, s: pd.Series):
        return s.map(lambda x: self.mapping.get(x, -1))

    def fit_transform(self, s: pd.Series):
        return self.fit(s).transform(s)

    def inverse_transform(self, s: pd.Series):
        return s.map(lambda x: self.inverse_mapping.get(x, None))

class OneHotEncoder:
    def __init__(self, drop_first=True, max_categories=20):
        self.drop_first = drop_first
        self.max_categories = max_categories
        self.categories = []

    def fit(self, s: pd.Series):
        top_cats = s.value_counts().index.tolist()
        self.categories = top_cats[:self.max_categories]
        if self.drop_first and len(self.categories) > 1:
            self.categories = self.categories[1:]
        return self

    def transform(self, s: pd.Series, prefix: str):
        df_encoded = pd.DataFrame(index=s.index)
        for cat in self.categories:
            df_encoded[f"{prefix}_{cat}"] = (s == cat).astype(int)
        return df_encoded

    def fit_transform(self, s: pd.Series, prefix: str):
        return self.fit(s).transform(s, prefix)

class TargetEncoder:
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = 0.0

    def fit(self, s: pd.Series, target: pd.Series):
        self.global_mean = float(target.mean())
        stats = target.groupby(s).agg(["count", "mean"])
        counts = stats["count"]
        means = stats["mean"]
        smooth = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
        self.mapping = smooth.to_dict()
        return self

    def transform(self, s: pd.Series):
        return s.map(lambda x: self.mapping.get(x, self.global_mean))

    def fit_transform(self, s: pd.Series, target: pd.Series):
        return self.fit(s, target).transform(s)

class FrequencyEncoder:
    def __init__(self):
        self.mapping = {}

    def fit(self, s: pd.Series):
        self.mapping = s.value_counts(normalize=True).to_dict()
        return self

    def transform(self, s: pd.Series):
        return s.map(lambda x: self.mapping.get(x, 0.0))

    def fit_transform(self, s: pd.Series):
        return self.fit(s).transform(s)

class BinaryEncoder:
    def __init__(self):
        self.label_map = {}
        self.n_bits = 0

    def fit(self, s: pd.Series):
        uniques = sorted(s.dropna().unique().tolist())
        self.label_map = {val: idx for idx, val in enumerate(uniques)}
        self.n_bits = int(np.ceil(np.log2(max(2, len(uniques) + 1))))
        return self

    def transform(self, s: pd.Series, prefix: str):
        mapped = s.map(lambda x: self.label_map.get(x, 0)).values
        df_res = pd.DataFrame(index=s.index)
        for bit in range(self.n_bits):
            df_res[f"{prefix}_bit_{bit}"] = ((mapped >> bit) & 1).astype(int)
        return df_res

    def fit_transform(self, s: pd.Series, prefix: str):
        return self.fit(s).transform(s, prefix)

class OrdinalEncoder:
    def __init__(self, order_dict=None):
        self.order_dict = order_dict or {}

    def fit(self, s: pd.Series):
        if not self.order_dict:
            uniques = sorted(s.dropna().unique().tolist())
            self.order_dict = {val: idx for idx, val in enumerate(uniques)}
        return self

    def transform(self, s: pd.Series):
        return s.map(lambda x: self.order_dict.get(x, -1))

    def fit_transform(self, s: pd.Series):
        return self.fit(s).transform(s)
