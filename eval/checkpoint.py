from torch import save, load
from torch.nn import Module
import os


def save_model(model: Module, name: str, epoch: int, path: str):
    """ Save a model to a file. """
    save(model.state_dict(),
         os.path.join(path, f"{name}-{epoch}.pth"))


def load_model(model: Module, name: str,  epoch: int, path: str):
    """ Load a model from a file. """
    model.load_state_dict(
        load(os.path.join(path, f"{name}-{epoch}.pth"), weights_only=False))
