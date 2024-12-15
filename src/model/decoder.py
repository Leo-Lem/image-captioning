from os.path import exists
from pandas import DataFrame
from torch import Tensor, load, save
from torch.nn import Module, Embedding, Linear
from torch.optim import Optimizer

from __param__ import DEBUG, PATHS, DATA, VOCAB, MODEL


class Decoder(Module):
    """ Base class for decoders with common functionality for embedding and dropout. """

    def __init__(self) -> None:
        super().__init__()
        self.image_fc = Linear(in_features=DATA.FEATURE_DIM,
                               out_features=MODEL.HIDDEN_DIM)
        self.embedding = Embedding(num_embeddings=VOCAB.SIZE,
                                   embedding_dim=MODEL.EMBEDDING_DIM,
                                   padding_idx=VOCAB.PADDING)
        self.fc = Linear(in_features=MODEL.HIDDEN_DIM,
                         out_features=VOCAB.SIZE)

    def forward(self, image: Tensor) -> Tensor:
        """ Forward pass for the decoder model that generates a sequence of tokens. """
        raise NotImplementedError("Subclasses should implement this!")

    best_model_path = PATHS.MODEL(f"{MODEL.NAME}-best.pt")
    model_path = PATHS.MODEL(f"{MODEL.NAME}.pt")
    losses_path = PATHS.OUT(f"losses-{MODEL.NAME}.csv")

    def load(self, optimizer: Optimizer = None, best: bool = False) -> DataFrame:
        """ Load the parameters from the disk if it exists and return the current epoch, the training losses and validation losses. """
        m_path = self.best_model_path if best else self.model_path
        if exists(m_path):
            DEBUG(f"Loading model from {m_path}…")

            checkpoint = load(m_path, weights_only=False)
            self.load_state_dict(checkpoint["state"])
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])
            return checkpoint["losses"]
        else:
            return DataFrame(columns=["Train", "Val"])

    def save(self, optimizer: Optimizer, losses: DataFrame) -> None:
        """ Save the parameters to the disk. Train and val losses are per epoch (index). """
        DEBUG(
            f"Saving model to {self.model_path} and losses to {self.losses_path}")

        save({
            "losses": losses,
            "state": self.state_dict(),
            "optimizer": optimizer.state_dict()
        }, self.model_path)

        if not exists(self.best_model_path) or losses["Val"].iloc[-1] < losses["Val"].min():
            save({
                "losses": losses,
                "state": self.state_dict(),
                "optimizer": optimizer.state_dict()
            }, self.best_model_path)

        losses.to_csv(self.losses_path, index=True)
