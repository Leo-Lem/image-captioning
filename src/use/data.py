# output data heads to the console (data-full.csv, vocab.csv, losses.csv, metrics.csv)

from pandas import read_csv

from __param__ import PATHS, MODEL


def print_data():
    print("======================================== Data ========================================")
    print(read_csv(PATHS.OUT("data-full.csv")).tail(), end="\n\n")

    print("======================================== Vocab =======================================")
    print(read_csv(PATHS.OUT("vocab.csv")).tail(), end="\n\n")

    print("======================================== Losses ======================================")
    print(read_csv(PATHS.OUT(f"losses-{MODEL.NAME}.csv")).tail(), end="\n\n")

    print("======================================== Metrics =====================================")
    print(read_csv(PATHS.OUT("metrics.csv")).tail(), end="\n\n")
