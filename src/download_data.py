"""Download the public Olist dataset and copy the CSV files into data/raw."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from src.config import RAW_DIR


def main() -> None:
    source = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    files = sorted(source.glob("*.csv"))
    if len(files) != 9:
        raise RuntimeError(f"Expected 9 CSV files, found {len(files)} at {source}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, RAW_DIR / file.name)
    print(f"Copied {len(files)} files to {RAW_DIR}")


if __name__ == "__main__":
    main()

