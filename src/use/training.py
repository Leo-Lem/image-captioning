from matplotlib import pyplot
from pandas import read_csv

from __param__ import PATHS, MODEL


def plot_training():
    """ Plot the training and validation loss over epochs. """
    data = read_csv(PATHS.OUT(f"losses-{MODEL.NAME}.csv"))

    pyplot.plot(data["Train"], label="Train")
    pyplot.plot(data["Val"], label="Val")

    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.title("Training")
    pyplot.savefig(PATHS.OUT(f"training-{MODEL.NAME}.png"))
    pyplot.close()
