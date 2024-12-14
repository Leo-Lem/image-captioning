from os import path
from pandas import DataFrame

from __param__ import PATHS


def store_results(results: DataFrame):
    """ Store the results in a CSV file. """
    results.to_csv(path.join(PATHS.OUT, "results.csv"), index=False)
