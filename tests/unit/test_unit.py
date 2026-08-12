import pytest
import os
import pandas as pd

def test_dummy_data_exists():
    dummy_csv_path = os.path.join(os.path.dirname(__file__), '..', 'dummy_data', 'dummy_data.csv')
    assert os.path.exists(dummy_csv_path)

def test_dummy_data_content():
    dummy_csv_path = os.path.join(os.path.dirname(__file__), '..', 'dummy_data', 'dummy_data.csv')
    if os.path.exists(dummy_csv_path):
        df = pd.read_csv(dummy_csv_path)
        assert len(df) == 5
        assert 'value' in df.columns
