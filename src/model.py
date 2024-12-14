from logging import debug as DEBUG
from os import path, makedirs
from torch import Tensor, cat, zeros, load, save
from torch.nn import Module, Embedding, Dropout, LSTM, Linear, GRU, Transformer, Parameter
from torch.optim import Optimizer

from __param__ import MODEL, DATA, APPROACH, PATHS, TRAIN


class Decoder(Module):
    """ Base class for decoders with common functionality for embedding and dropout. """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings=DATA.VOCAB_SIZE,
                                   embedding_dim=MODEL.EMBEDDING_DIM,
                                   padding_idx=DATA.PADDING)
        self.fc = Linear(in_features=MODEL.HIDDEN_DIM,
                         out_features=DATA.VOCAB_SIZE)

    @classmethod
    def create(self) -> "Decoder":
        """ Create a new decoder model based on the specified approach. """
        if APPROACH == "gru":
            return GRUDecoder()
        elif APPROACH == "lstm":
            return LSTMDecoder()
        elif APPROACH == "transformer":
            return TransformerDecoder()
        else:
            raise ValueError(f"Unknown approach: {APPROACH}")

    def forward(self, image: Tensor) -> Tensor:
        """ Forward pass for the decoder from image to caption. """
        raise NotImplementedError("Subclasses should implement this!")

    # --- Model saving and loading ---
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

        makedirs(self.model_path, exist_ok=True)
        save({
            "epoch": epoch+1,
            "state": self.state_dict(),
            "optimizer": optimizer.state_dict()
        }, self.best_model_path if is_best else self.model_path)


class GRUDecoder(Decoder):
    """ GRU-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        self.gru = GRU(input_size=DATA.CAPTION_LEN,
                       hidden_size=MODEL.HIDDEN_DIM,
                       num_layers=MODEL.NUM_LAYERS,
                       dropout=MODEL.DROPOUT,
                       batch_first=True)

    def forward(self, image: Tensor, caption: Tensor) -> Tensor:
        """ Forward pass for the GRU decoder. """
        # (batch_size, caption_len, embedding_dim)
        embeddings = self.embedding(caption)
        embeddings = cat((image.unsqueeze(1), embeddings), dim=1)
        # (batch_size, seq_len+1, hidden_dim)
        outputs, _ = self.gru(embeddings)
        # (batch_size, seq_len+1, vocab_size)
        outputs = self.fc(outputs)

        return outputs


class LSTMDecoder(Decoder):
    """ LSTM-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = LSTM(input_size=MODEL.EMBEDDING_DIM,
                         hidden_size=MODEL.HIDDEN_DIM,
                         num_layers=MODEL.NUM_LAYERS,
                         dropout=MODEL.DROPOUT,
                         batch_first=True)

    def forward(self, image: Tensor, caption: Tensor) -> Tensor:
        """ Forward pass for the LSTM decoder. """
        assert image.size() == (TRAIN.BATCH_SIZE, DATA.FEATURE_DIM)
        assert caption.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN)

        embedding = self.embedding(caption)
        assert embedding.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN, MODEL.EMBEDDING_DIM)

        embedding = cat((image.unsqueeze(1), embedding), dim=1)
        assert embedding.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN+1, DATA.FEATURE_DIM)

        output, _ = self.lstm(embedding)
        assert output.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN+1, MODEL.HIDDEN_DIM)

        output = self.fc(output)
        assert output.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN+1, DATA.VOCAB_SIZE)

        return output


class TransformerDecoder(Decoder):
    """ Transformer-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        max_length = 100
        self.pos_enc = Parameter(zeros(1, max_length, DATA.CAPTION_LEN))
        self.transformer = Transformer(d_model=DATA.CAPTION_LEN,
                                       nhead=MODEL.NUM_HEADS,
                                       num_encoder_layers=0,
                                       num_decoder_layers=MODEL.NUM_LAYERS,
                                       dropout=MODEL.DROPOU,
                                       batch_first=True)

    def forward(self, image: Tensor, caption: Tensor) -> Tensor:
        """ Forward pass for the Transformer decoder. """
        # (batch_size, caption_Len, embedding_dim)
        embeddings = self.embedding(caption)
        # Add positional encoding
        embeddings += self.pos_enc[:, :caption.size(1), :]
        # Repeat features
        image = image\
            .unsqueeze(1)\
            .repeat(1, caption.size(1), 1)
        # Combine features and embeddings
        embeddings = cat((image, embeddings), dim=1)
        tgt_mask = Transformer\
            .generate_square_subsequent_mask(embeddings.size(1))\
            .to(embeddings.device)
        # (seq_len, batch_size, d_model)
        outputs = self.transformer(
            embeddings.permute(1, 0, 2), tgt_mask=tgt_mask)
        # (batch_size, seq_len, vocab_size)
        outputs = self.fc(outputs.permute(1, 0, 2))
        return outputs
