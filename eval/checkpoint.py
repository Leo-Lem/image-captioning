from torch import save, load
from torch.nn import Module


def save(model: Module, epoch: int, path: str):
    """ Save a model to a file. """
    save(model.state_dict(), f"{path}-{epoch}.pth")


def load(model: Module, epoch: int, path: str):
    """ Load a model from a file. """
    try:
        model.load_state_dict(load(f"{path}-{epoch}.pth"))
    except FileNotFoundError:
        print(f"Checkpoint {epoch} not found.")
        raise
