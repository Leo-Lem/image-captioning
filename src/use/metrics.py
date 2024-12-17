from matplotlib import pyplot
from pandas import read_csv
from numpy import arange
from tqdm import tqdm

from __param__ import PATHS


def plot_metrics(references: dict[str, float]):
    """ Plot the metrics of the models. """
    data = read_csv(PATHS.OUT("metrics.csv"))

    models = data["Model"]
    metrics = ["BLEU", "METEOR", "NIST"]

    for metric, value in references.items():
        pyplot.axhline(value, color="red", linestyle="--",
                       label=f"{metric} reference")

    bar_width = 0.2
    x = arange(len(metrics))

    for i, (_, row) in enumerate(data.iterrows()):
        bleu = row["BLEU"]
        meteor = row["METEOR"]
        nist = row["NIST"]
        pyplot.bar(x + i * bar_width,
                   [bleu, meteor, nist], width=bar_width, label=row["Model"])

    pyplot.xticks(x + (bar_width * (len(models) - 1)) / 2, metrics)
    pyplot.legend()
    pyplot.title("Results")
    pyplot.savefig(PATHS.OUT("metrics.png"))
    tqdm.write(f"Metrics saved to '{PATHS.OUT('metrics.png')}'.")
    pyplot.close()
