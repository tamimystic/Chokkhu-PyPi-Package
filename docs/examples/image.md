# End-to-End Image Pipeline

Below is a complete example of using Chokkhu to load, augment, and classify images.

```python
import chokkhu as ck
import numpy as np
from chokkhu.transformation import PCA

# 1. Load Image Folder
image_dir = "dataset/images/"
img_dict = ck.load(image_dir, type="image", img_size=(32, 32)) 

# 2. Image EDA
ck.eda.image(dataset_path=image_dir, save_reports=True)

# Format for Augmentation
img_dict["images"] = list(img_dict["X"])
img_dict["labels"] = list(img_dict["y"])

# 3. Data Augmentation
aug_img_dict = ck.transform(
    data=img_dict, 
    augment=True, 
    augment_techniques=["horizontal_flip", "rotate"], 
    augment_factor=1
)

# 4. Flatten Images
X_flattened = np.array([img.flatten() for img in aug_img_dict["images"]])
y_labels = np.array(aug_img_dict["labels"])

# 5. Dimensionality Reduction (PCA)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_flattened)

data_for_split = {"X": X_pca, "y": y_labels}

# 6. Split
X_train, X_test, y_train, y_test = ck.split(data_for_split, test_size=0.2, stratify=True)

# 7. Train Model (KNN)
image_model = ck.train(
    model="knn", 
    X_train=X_train, 
    y_train=y_train, 
    task="classification", 
    n_neighbors=3
)

# 8. Evaluate
results = ck.evaluate(image_model, X_test, y_test, save_reports=True)
print(results)
```
