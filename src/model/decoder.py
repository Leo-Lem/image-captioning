from logging import debug as DEBUG
from os import path, makedirs
from torch import Tensor, load, save
from torch.nn import Module, Embedding, Linear
from torch.optim import Optimizer

from __param__ import APPROACH, MODEL, DATA, PATHS


class Decoder(Module):
    """ Base class for decoders with common functionality for embedding and dropout. """

    def __init__(self) -> None:
        super().__init__()
        self.image_fc = Linear(in_features=DATA.FEATURE_DIM,
                               out_features=MODEL.EMBEDDING_DIM)
        self.embedding = Embedding(num_embeddings=DATA.VOCAB_SIZE,
                                   embedding_dim=MODEL.EMBEDDING_DIM,
                                   padding_idx=DATA.PADDING)
        self.fc = Linear(in_features=MODEL.HIDDEN_DIM,
                         out_features=DATA.VOCAB_SIZE)

    def forward(self, image: Tensor) -> Tensor:
        """ Forward pass for the decoder model that generates a sequence of tokens. """
        raise NotImplementedError("Subclasses should implement this!")

    model_name = f"{APPROACH}-model"
    best_model_path = path.join(PATHS.MODEL, f"{model_name}-best.pt")
    model_path = path.join(PATHS.MODEL, f"{model_name}.pt")

    def load(self, optimizer: Optimizer) -> int:
        """ Load the parameters from the disk if it exists and return the current epoch. """
        if path.exists(self.model_path):
            DEBUG(f"Loading model from {self.model_path}…")

            checkpoint = load(self.model_path)
            self.load_state_dict(checkpoint["state"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            return checkpoint["epoch"]
        else:
            return 0

    def save(self, optimizer: Optimizer, epoch: int, is_best: bool):
        """ Save the parameters to the disk. """
        DEBUG(f"Saving model to {self.model_path}…")

        makedirs(path.dirname(self.model_path), exist_ok=True)
        save({
            "epoch": epoch+1,
            "state": self.state_dict(),
            "optimizer": optimizer.state_dict()
        }, self.best_model_path if is_best else self.model_path)
