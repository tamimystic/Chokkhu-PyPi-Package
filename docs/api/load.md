# Loading Data (ck.load)

The `ck.load()` function auto-detects the file format and loads tabular or image datasets seamlessly into memory.

## Parameters Configuration

- **Default usage:** `ck.load("dataset.csv")`
- **Strict Parameters:**
  - `path` (str): The absolute or relative path to the dataset file or directory.
- **Dynamic Parameters (Changeable):**
  - `type` (str): Default `"auto"`. Options: `"auto"`, `"tabular"`, `"image"`.
  - `img_size` (tuple): Default `(64, 64)`. The dimensions to resize images to if `type="image"` is specified.
