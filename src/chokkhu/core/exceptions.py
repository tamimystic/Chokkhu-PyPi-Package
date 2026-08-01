class ChokkhuError(Exception):
    """Base exception class for Chokkhu package."""

    pass


class DataLoadError(ChokkhuError):
    """Raised when there is an issue loading the dataset."""

    pass


class InvalidFormatError(ChokkhuError):
    """Raised when an unsupported file format is provided."""

    pass
