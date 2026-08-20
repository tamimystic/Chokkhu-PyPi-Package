import os


def test_integration_dummy():
    dummy_csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dummy_data", "dummy_data.csv")
    )
    assert os.path.exists(dummy_csv_path)
