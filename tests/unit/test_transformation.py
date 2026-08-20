from __future__ import annotations

import numpy as np
import pandas as pd

import chokkhu
from chokkhu.transformation import (
    ADASYN,
    LDA,
    PCA,
    SMOTE,
    TSNE,
    BinningTransformer,
    ImageAugmenter,
    LogTransformer,
    PolynomialFeatures,
    RandomOverSampler,
    RandomUnderSampler,
    SMOTETomek,
    TruncatedSVD,
)


def test_pca_and_svd():
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    assert X_pca.shape == (4, 2)
    assert pca.explained_variance_ratio_ is not None
    assert len(pca.explained_variance_ratio_) == 2

    X_inv = pca.inverse_transform(X_pca)
    assert X_inv.shape == (4, 3)

    pca_var = PCA(variance_ratio=0.9)
    X_var = pca_var.fit_transform(X)
    assert X_var.shape[0] == 4

    svd = TruncatedSVD(n_components=2)
    X_svd = svd.fit_transform(X)
    assert X_svd.shape == (4, 2)


def test_lda():
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [6.0, 9.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X, y)
    assert X_lda.shape == (4, 1)


def test_tsne():
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [5.0, 6.0, 7.0],
            [5.1, 6.1, 7.1],
        ]
    )
    tsne = TSNE(n_components=2, n_iter=50, perplexity=2.0, random_state=42)
    X_tsne = tsne.fit_transform(X)
    assert X_tsne.shape == (4, 2)


def test_resampling_methods():
    X = np.array(
        [
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3],
            [10.0, 20.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 1])

    ros = RandomOverSampler(ratio=1.0, random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y)
    assert len(y_ros) == 8

    rus = RandomUnderSampler(ratio=1.0, random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    assert len(y_rus) == 2

    smote = SMOTE(k_neighbors=1, ratio=1.0, random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)
    assert len(y_sm) >= 5

    adasyn = ADASYN(k_neighbors=2, ratio=1.0, random_state=42)
    X_ada, y_ada = adasyn.fit_resample(X, y)
    assert len(y_ada) >= 5

    sm_tom = SMOTETomek(k_neighbors=1, ratio=1.0, random_state=42)
    X_tom, y_tom = sm_tom.fit_resample(X, y)
    assert len(y_tom) >= 2


def test_image_augmenter():
    img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    augmenter = ImageAugmenter(augment_factor=2, random_state=42)
    augmented = augmenter.augment_image(img)
    assert len(augmented) == 2
    assert augmented[0].shape == (64, 64, 3)

    imgs, lbls = augmenter.augment_dataset([img], labels=["cat"])
    assert len(imgs) == 3
    assert len(lbls) == 3


def test_feature_engineering():
    X = np.array([[2.0, 3.0], [4.0, 5.0]])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    assert X_poly.shape == (2, 5)

    df = pd.DataFrame({"a": [1.0, 10.0, 100.0], "b": [5.0, 15.0, 25.0]})
    log_df = LogTransformer().fit_transform(df)
    assert pd.api.types.is_numeric_dtype(log_df["a"].dtype)

    bin_df = BinningTransformer(n_bins=2).fit_transform(df)
    assert bin_df["a"].nunique() <= 2


def test_unified_transform():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 10.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 100.0],
            "cat": ["a", "a", "a", "a", "b"],
            "target": [0, 0, 0, 0, 1],
        }
    )
    res_pca = chokkhu.transform(df, target="target", pca=1)
    assert "pca_0" in res_pca.columns

    res_smote = chokkhu.transform(df, target="target", resample="smote", smote_k=1)
    assert len(res_smote) >= 5

    res_poly = chokkhu.transform(df, target="target", polynomial=2)
    assert "poly_0" in res_poly.columns

    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    data_img = {"images": [img], "labels": ["dog"]}
    res_img = chokkhu.transform(data_img, augment=True, augment_factor=1)
    assert len(res_img["images"]) == 2
