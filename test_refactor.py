import sys
sys.path.insert(0, "src")
import chokkhu as ck
import pandas as pd
import numpy as np

# Create dummy dataframe
np.random.seed(42)
df = pd.DataFrame({
    'CG': np.random.uniform(2.5, 4.0, 100),
    'Marks': np.random.uniform(50, 100, 100),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Year': np.random.choice(['1st', '2nd', '3rd', '4th'], 100)
})
df.to_csv("test_dummy.csv", index=False)

try:
    ck.eda.tabular(
        dataset_path="test_dummy.csv",
        save_reports=True,
        save_dir="test_reports",
        target_col="CG"
    )
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
