from logging import debug as DEBUG
from pandas import DataFrame, read_csv
from typing import Literal

from __param__ import PATHS, DATA


def load_data(name: Literal["full", "train", "val", "test"]) -> DataFrame:
    """ Load the specified dataset. """
    if DATA.SAMPLE:
        name = "sample"

    data = read_csv(f"{PATHS.OUT}/data-{name}.csv", dtype=str)

    DEBUG(f"Loaded {name} data ({data.shape})\n{data.head(3)}")

    return data
