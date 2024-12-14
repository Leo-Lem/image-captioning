from logging import debug as DEBUG
from os import path, makedirs
from torch import Tensor, zeros, load, save, full, long, stack, triu, ones
from torch.nn import Module, Embedding, LSTM, Linear, GRU, Parameter, TransformerDecoderLayer
from torch.nn import TransformerDecoder as TransformerDecoderTorch
from torch.optim import Optimizer

from __param__ import MODEL, DATA, APPROACH, PATHS, TRAIN


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
        """ Forward pass for the decoder model that generates a sequence of tokens. """
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

        makedirs(path.dirname(self.model_path), exist_ok=True)
        save({
            "epoch": epoch+1,
            "state": self.state_dict(),
            "optimizer": optimizer.state_dict()
        }, self.best_model_path if is_best else self.model_path)


class GRUDecoder(Decoder):
    """ GRU-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        self.gru = GRU(input_size=MODEL.EMBEDDING_DIM,
                       hidden_size=MODEL.HIDDEN_DIM,
                       num_layers=MODEL.NUM_LAYERS,
                       dropout=MODEL.DROPOUT,
                       batch_first=True)

    def forward(self, image: Tensor) -> Tensor:
        """ Predict the caption for the given image. """
        assert image.size() == (TRAIN.BATCH_SIZE, 1, DATA.FEATURE_DIM)

        hidden = None
        input = full((TRAIN.BATCH_SIZE, 1), DATA.START,
                     dtype=long, device=image.device)
        embedding = self.image_fc(image)
        outputs = []

        for _ in range(DATA.CAPTION_LEN):
            output, hidden = self.gru(embedding, hidden)
            assert output.size() == (TRAIN.BATCH_SIZE, 1, MODEL.HIDDEN_DIM)

            output = self.fc(output.squeeze(1))
            assert output.size() == (TRAIN.BATCH_SIZE, DATA.VOCAB_SIZE)

            outputs.append(output)
            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (TRAIN.BATCH_SIZE, 1)

            embedding = self.embedding(input)
            assert embedding.size() == (TRAIN.BATCH_SIZE, 1, MODEL.EMBEDDING_DIM)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN, DATA.VOCAB_SIZE)

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

    def forward(self, image: Tensor) -> Tensor:
        """ Predict the caption for the given image. """
        assert image.size() == (TRAIN.BATCH_SIZE, 1, DATA.FEATURE_DIM)

        hidden = None
        input = full((TRAIN.BATCH_SIZE, 1),
                     DATA.START,
                     dtype=long,
                     device=image.device)
        embedding = self.image_fc(image)
        outputs = []

        for _ in range(DATA.CAPTION_LEN):
            output, hidden = self.lstm(embedding, hidden)
            assert output.size() == (TRAIN.BATCH_SIZE, 1, MODEL.HIDDEN_DIM)

            output = self.fc(output.squeeze(1))
            assert output.size() == (TRAIN.BATCH_SIZE, DATA.VOCAB_SIZE)

            outputs.append(output)
            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (TRAIN.BATCH_SIZE, 1)

            embedding = self.embedding(input)
            assert embedding.size() == (TRAIN.BATCH_SIZE, 1, MODEL.EMBEDDING_DIM)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN, DATA.VOCAB_SIZE)

        return outputs


class TransformerDecoder(Decoder):
    """ Transformer-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        max_length = 100
        self.pos_enc = Parameter(zeros(1, max_length, MODEL.EMBEDDING_DIM))
        self.transformer_decoder = TransformerDecoderTorch(
            TransformerDecoderLayer(d_model=MODEL.EMBEDDING_DIM,
                                    nhead=MODEL.ATTENTION_HEADS,
                                    dim_feedforward=MODEL.HIDDEN_DIM,
                                    dropout=MODEL.DROPOUT,
                                    batch_first=True),
            num_layers=MODEL.NUM_LAYERS)
        self.fc = Linear(in_features=MODEL.EMBEDDING_DIM,
                         out_features=DATA.VOCAB_SIZE)

    def forward(self, image: Tensor) -> Tensor:
        """ Predict the caption for the given image. """
        assert image.size() == (TRAIN.BATCH_SIZE, 1, DATA.FEATURE_DIM)

        input = full((TRAIN.BATCH_SIZE, 1),
                     DATA.START,
                     dtype=long,
                     device=image.device)
        memory = self.image_fc(image)
        memory += self.pos_enc[:, :1, :]
        outputs = []

        for t in range(DATA.CAPTION_LEN):
            tgt = self.embedding(input) + self.pos_enc[:, :t + 1, :]
            tgt_mask = self.mask(t + 1).to(image.device)

            decoder_output = self.transformer_decoder(tgt=tgt,
                                                      memory=memory,
                                                      tgt_mask=tgt_mask
                                                      )
            output = self.fc(decoder_output[:, -1, :])
            assert output.size() == (TRAIN.BATCH_SIZE, DATA.VOCAB_SIZE)

            outputs.append(output)
            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (TRAIN.BATCH_SIZE, 1)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN, DATA.VOCAB_SIZE)

        return outputs

    @staticmethod
    def mask(size: int) -> Tensor:
        mask = triu(ones(size, size), diagonal=1).bool()
        return mask
