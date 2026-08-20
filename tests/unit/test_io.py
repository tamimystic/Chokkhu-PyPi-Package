import os
import tempfile
import numpy as np
import pandas as pd
import cv2
import pytest
import chokkhu

def test_load_and_save_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
        csv_path = os.path.join(tmpdir, "test.csv")
        saved_path = chokkhu.save(df, csv_path)
        assert os.path.exists(saved_path)
        loaded_df = chokkhu.load(csv_path)
        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (3, 3)
        assert list(loaded_df.columns) == ["a", "b", "c"]

def test_load_and_save_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({"num": [10, 20], "text": ["hello", "world"]})
        json_path = os.path.join(tmpdir, "test.json")
        chokkhu.save(df, json_path)
        assert os.path.exists(json_path)
        loaded = chokkhu.load(json_path)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 2

def test_load_and_save_numpy():
    with tempfile.TemporaryDirectory() as tmpdir:
        arr = np.array([[1, 2], [3, 4]])
        npy_path = os.path.join(tmpdir, "test.npy")
        chokkhu.save(arr, npy_path)
        assert os.path.exists(npy_path)
        loaded = np.load(npy_path)
        assert np.array_equal(arr, loaded)

def test_load_images_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        class_a = os.path.join(tmpdir, "cats")
        class_b = os.path.join(tmpdir, "dogs")
        os.makedirs(class_a)
        os.makedirs(class_b)
        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(class_a, "cat1.jpg"), dummy_img)
        cv2.imwrite(os.path.join(class_b, "dog1.png"), dummy_img)
        res = chokkhu.load(tmpdir, type="image", img_size=(32, 32), normalize=True)
        assert isinstance(res, dict)
        assert "X" in res and "y" in res and "class_names" in res
        assert len(res["X"]) == 2
        assert res["X"][0].shape == (32, 32, 3)
        assert res["class_names"] == ["cats", "dogs"]
