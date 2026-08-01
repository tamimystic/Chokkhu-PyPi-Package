import sys

from tqdm import tqdm


class Logger:
    @staticmethod
    def info(msg: str) -> None:
        """Prints a standardized info log message."""
        print(f"\n[INFO] {msg}")
        sys.stdout.flush()

    @staticmethod
    def error(msg: str) -> None:
        """Prints a standardized error log message."""
        print(f"\n[ERROR] {msg}")
        sys.stdout.flush()

    @staticmethod
    def warning(msg: str) -> None:
        """Prints a standardized warning log message."""
        print(f"\n[WARNING] {msg}")
        sys.stdout.flush()


def get_progress_bar(iterable, desc: str):
    """Returns a tqdm progress bar with standardized settings."""
    return tqdm(iterable, desc=desc, leave=False)
