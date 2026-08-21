# Loading Data (`ck.load`)

The `ck.load()` function auto-detects the file format and loads tabular or image datasets seamlessly into memory.

## Syntax

```python
import chokkhu as ck

df = ck.load("dataset.csv")
```

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | Required | The absolute or relative path to the dataset file or directory. |
| `type` | `str` | `"auto"` | Forces the loader to treat data as a specific type. Options: `"auto"`, `"tabular"`, `"image"`. |
| `img_size` | `tuple` | `(64, 64)` | The dimensions to resize images to if `type="image"` is specified. |

??? example "View Image Loading Example"
    ```python
    # Load an image directory containing class subfolders
    images_dict = ck.load(
        "images_folder/", 
        type="image", 
        img_size=(128, 128)
    )
    
    # Access features and labels
    X = images_dict["X"]
    y = images_dict["y"]
    ```
