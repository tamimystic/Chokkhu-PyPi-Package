from __future__ import annotations

import chokkhu


def test_package_metadata():
    assert hasattr(chokkhu, "__version__")
    assert isinstance(chokkhu.__version__, str)


def test_package_exports():
    assert hasattr(chokkhu, "clean")
    assert hasattr(chokkhu, "preprocess")
    assert hasattr(chokkhu, "split")
    assert hasattr(chokkhu, "load")
    assert hasattr(chokkhu, "save")
    assert hasattr(chokkhu, "eda")
