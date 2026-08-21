# Data Splitting (ck.split)

Chokkhu provides multi-way stratified partitioning and cross-validation generators.

## Parameters Configuration

- **Default usage:** `ck.split(df, target="price", test_size=0.2)`
- **Strict Parameters:**
  - `data` (pd.DataFrame or dict): The input data structure.
  - `target` (str): The target column or key.
- **Dynamic Parameters (Changeable):**
  - `method` (str): Default `"train_test"`. Options: `"train_test"`, `"kfold"`, `"stratified_kfold"`, `"timeseries"`.
  - `test_size` (float): Default `0.2`. The proportion of the dataset to include in the test split.
  - `val_size` (float): Default `0.0`. If greater than 0.0, a 3-way split (Train/Val/Test) is returned.
  - `stratify` (bool): Default `False`. If True, ensures class distribution is maintained in splits.
  - `random_state` (int): Default `None`. Seed for reproducibility.
  - `n_splits` (int): Default `5`. Only used if `method` is `"kfold"` or `"stratified_kfold"`.
