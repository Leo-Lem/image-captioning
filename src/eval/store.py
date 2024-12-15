from os.path import exists
from pandas import DataFrame

from __param__ import PATHS


def store_results(results: DataFrame):
    """ Store the results in a CSV file. """
    file = PATHS.OUT("results.csv")
    results.to_csv(file, index=True, mode="a", header=not exists(file))
