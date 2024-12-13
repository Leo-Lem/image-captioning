from os import path
from torch import load
from torch.optim import Optimizer

from .decoder import Decoder
from __param__ import APPROACH, PATHS, FLAGS

model_name = f"{APPROACH}-model"
best_model_path = path.join(PATHS.MODEL, f"{model_name}-best.pt")
model_path = path.join(PATHS.MODEL, f"{model_name}.pt")


def load_model(model: Decoder, optimizer: Optimizer) -> int:
    """
    Load the parameters from the disk if it exists.

    :return: The current epoch.
    """
    if path.exists(model_path):
        if FLAGS.DEBUG:
            print(f"\tLoading model from {model_path}…")

        checkpoint = load(model_path)
        model.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"]
    else:
        return 0
