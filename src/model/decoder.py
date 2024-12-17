from os.path import exists
from pandas import DataFrame
from torch import Tensor, load, save, multinomial, softmax, stack, full
from torch.nn import Module, Embedding, Linear
from torch.optim import Optimizer

from __param__ import DEBUG, PATHS, DATA, MODEL, TRAIN
from src.data import Vocabulary


class Decoder(Module):
    """ Base class for decoders with common functionality for embedding and dropout. """

    def __init__(self) -> None:
        super().__init__()
        self.image_to_hidden_fc = Linear(in_features=DATA.FEATURE_DIM,
                                         out_features=MODEL.HIDDEN_DIM)
        self.indices_to_embeddings = Embedding(num_embeddings=Vocabulary.SIZE,
                                               embedding_dim=MODEL.EMBEDDING_DIM,
                                               padding_idx=Vocabulary.PADDING)
        self.hidden_to_logits_fc = Linear(in_features=MODEL.HIDDEN_DIM,
                                          out_features=Vocabulary.SIZE)

    def forward(self, images: Tensor, caption: Tensor = None) -> Tensor:
        """ Forward pass for the decoder model that generates a sequence of tokens with optional teacher forcing. """
        raise NotImplementedError("Forward pass not implemented.")

    def predict(self, images: Tensor) -> Tensor:
        """ Predict a sequence of indices for the given image. """
        self.eval()
        indices = self._predict_indices(self(images))
        return indices

    def _predict_indices(self, logits: Tensor) -> Tensor:
        """ Predict the token indices from the logits using softmax sampling. """
        assert logits.size() == (logits.size(0), DATA.CAPTION_LEN-1, Vocabulary.SIZE)
        indices = stack([self._predict_index(logit)
                        for logit in logits], dim=1).transpose(0, 1)
        assert indices.size(1) <= DATA.CAPTION_LEN
        return indices

    def _predict_index(self, logit: Tensor) -> Tensor:
        """ Predict the next token index from the logits using softmax sampling. """
        assert logit.size() == (logit.size(0), Vocabulary.SIZE)
        index = multinomial(softmax(logit, dim=-1), num_samples=1).squeeze(-1)
        assert index.size() == (logit.size(0),)
        return index

    def _validate(self, images: Tensor, caption: Tensor) -> int:
        """ Validate the input tensors for the forward pass and retrieve the batch size. """
        batch_size = images.size(0)
        assert caption is None or caption.size() == (batch_size, DATA.CAPTION_LEN)
        return batch_size

    def _start_index(self, batch_size: int) -> Tensor:
        index: Tensor = \
            full((batch_size,), fill_value=Vocabulary.START, device=TRAIN.DEVICE)
        assert index.size() == (batch_size,)
        return index

    def _image_to_hidden(self, image: Tensor, batch_size: int) -> Tensor:
        assert image.size() == (batch_size, DATA.FEATURE_DIM), image.size()
        hidden: Tensor = self.image_to_hidden_fc(image.unsqueeze(0))
        hidden = hidden.squeeze(1).repeat(MODEL.NUM_LAYERS, 1, 1)
        assert hidden.size() == (MODEL.NUM_LAYERS, batch_size, MODEL.HIDDEN_DIM)
        return hidden

    def _validate_prediction(self, logits: list[Tensor]) -> Tensor:
        """ Validate the prediction tensor. """
        prediction: Tensor = stack(logits, dim=1)
        assert prediction.size() == \
            (prediction.size(0), DATA.CAPTION_LEN-1, Vocabulary.SIZE)
        return prediction

    best_model_path = PATHS.MODEL(f"{MODEL.NAME}-best.pt")
    model_path = PATHS.MODEL(f"{MODEL.NAME}.pt")
    losses_path = PATHS.OUT(f"losses-{MODEL.NAME}.csv")

    def load(self, optimizer: Optimizer = None, best: bool = False) -> DataFrame:
        """ Load the parameters from the disk if it exists and return the current epoch, the training losses and validation losses. """
        m_path = self.best_model_path if best else self.model_path
        if exists(m_path):
            checkpoint = load(m_path, weights_only=False)
            self.load_state_dict(checkpoint["state"])
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])
            return checkpoint["losses"]
        else:
            return DataFrame(columns=["Training", "Validation"])

    def save(self, optimizer: Optimizer, losses: DataFrame) -> None:
        """ Save the parameters to the disk. Train and val losses are per epoch (index). """
        save({
            "losses": losses,
            "state": self.state_dict(),
            "optimizer": optimizer.state_dict()
        }, self.model_path)

        if not exists(self.best_model_path) or losses["Validation"].iloc[-1] < losses["Validation"].min():
            save({
                "losses": losses,
                "state": self.state_dict(),
                "optimizer": optimizer.state_dict()
            }, self.best_model_path)

        losses.to_csv(self.losses_path, index=False)
