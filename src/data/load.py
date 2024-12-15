from pandas import DataFrame, read_csv
from typing import Literal

from __param__ import PATHS, DATA, DEBUG


def load_data(name: Literal["full", "train", "val", "test"]) -> DataFrame:
    """ Load the specified dataset. """
    name = 'sample' if DATA.SAMPLE else name
    data = read_csv(PATHS.OUT(f"data-{name}.csv"), dtype=str)
    DEBUG(f"Loaded {name} data ({data.shape})\n{data.head(3)}")
    return data
