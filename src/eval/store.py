from os import path
from pandas import DataFrame

from __param__ import PATHS


def store_results(results: DataFrame):
    """ Store the results in a CSV file. """
    file = path.join(PATHS.OUT, "results.csv")
    results.to_csv(file, index=False, mode="a", header=not path.exists(file))
