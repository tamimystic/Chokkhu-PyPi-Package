# Loading Data (ck.load)

The ck.load() function auto-detects the file format and loads tabular or image datasets seamlessly into memory.

## Syntax

`python
import chokkhu as ck

# Load Tabular Data
df = ck.load("dataset.csv")

# Load Image Data
images_dict = ck.load("images_folder/", type="image", img_size=(128, 128))
`

## Supported Formats
- **Tabular**: .csv, .tsv, .xlsx, .json, .parquet
- **NumPy arrays**: .npy, .npz
- **Image Folders**: Pass a directory path containing subfolders of classes.
