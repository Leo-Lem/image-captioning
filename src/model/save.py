from os import path, makedirs
from torch import save
from torch.optim import Optimizer

from .decoder import Decoder
from __param__ import APPROACH, PATHS, FLAGS

model_name = f"{APPROACH}-model"
best_model_path = path.join(PATHS.MODEL, f"{model_name}-best.pt")
model_path = path.join(PATHS.MODEL, f"{model_name}.pt")


def save_model(epoch: int, model: Decoder, optimizer: Optimizer, is_best: bool):
    """
    Save the parameters to the disk.

    :param model: The model to save.
    """
    if FLAGS.DEBUG:
        print(f"\tSaving model to {model_path}…")

    makedirs(model_path, exist_ok=True)
    save({
        "epoch": epoch+1,
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, best_model_path if is_best else model_path)
