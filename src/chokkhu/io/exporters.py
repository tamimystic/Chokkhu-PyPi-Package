import os
import pickle
import numpy as np
import pandas as pd
from chokkhu.core.logger import Logger

def save(data, path: str, format: str = "auto", verbose: bool = True, **kwargs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower() if format == "auto" else f".{format.lower()}"
    if isinstance(data, pd.DataFrame):
        savers = {
            ".csv": lambda d, p, **kw: d.to_csv(p, index=kw.get("index", False)),
            ".tsv": lambda d, p, **kw: d.to_csv(p, sep="\t", index=kw.get("index", False)),
            ".json": lambda d, p, **kw: d.to_json(p, orient=kw.get("orient", "records")),
            ".parquet": lambda d, p, **kw: d.to_parquet(p, index=kw.get("index", False)),
            ".xlsx": lambda d, p, **kw: d.to_excel(p, index=kw.get("index", False)),
            ".feather": lambda d, p, **kw: d.to_feather(p),
        }
        if ext in savers:
            savers[ext](data, path, **kwargs)
        else:
            data.to_csv(path, index=False)
    elif isinstance(data, np.ndarray):
        if ext == ".npy":
            np.save(path, data)
        elif ext == ".npz":
            np.savez_compressed(path, data=data)
        elif ext == ".csv":
            np.savetxt(path, data, delimiter=",")
        else:
            np.save(path, data)
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    if verbose:
        Logger.info(f"Saved dataset to {path}")
    return path
