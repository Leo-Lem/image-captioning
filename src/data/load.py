from logging import debug as DEBUG
from pandas import DataFrame, read_csv
from typing import Literal

from __param__ import PATHS, FLAGS


def load_data(name: Literal["full", "train", "val", "test", "sample"]) -> DataFrame:
    """ Load the specified dataset. """
    data = read_csv(f"{PATHS.RES}/{name}.csv", dtype=str)

    DEBUG(f"Loaded {name} data ({data.shape})\n{data.head(3)}")

    return data
