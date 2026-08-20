from __future__ import annotations

import sys

from tqdm import tqdm


class Logger:

    @staticmethod
    def info(msg: str) -> None:
        print(f"\n[INFO] {msg}")
        sys.stdout.flush()

    @staticmethod
    def error(msg: str) -> None:
        print(f"\n[ERROR] {msg}")
        sys.stdout.flush()

    @staticmethod
    def warning(msg: str) -> None:
        print(f"\n[WARNING] {msg}")
        sys.stdout.flush()


def get_progress_bar(iterable, desc: str):
    return tqdm(iterable, desc=desc, leave=False)
