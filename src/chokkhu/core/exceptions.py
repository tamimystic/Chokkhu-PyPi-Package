from __future__ import annotations


class ChokkhuError(Exception):
    pass


class DataLoadError(ChokkhuError):
    pass


class InvalidFormatError(ChokkhuError):
    pass
