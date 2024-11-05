from torch import Tensor, tensor, full, cat, softmax, float32
from torch.nn import Module, Embedding, Linear, GRUCell, LSTMCell
from data import Vocabulary


class CaptionDecoder(Module):
    """Abstract base class for decoders to generate captions."""

    def __init__(self, vocabulary_size: int, feature_size: int, embedding_size: int, hidden_size: int):
        """
        Args:
            vocabulary_size: The number of tokens in the vocabulary.
            embedding_size: The size of the token embeddings.
            hidden_size: The size of the hidden state of the RNN.
            feature_size: The size of the encoder's output features.
        """
        super().__init__()
        self.embedding_layer = Embedding(vocabulary_size, embedding_size)
        self.feature_projection = Linear(feature_size, hidden_size)
        self.output_layer = Linear(hidden_size, vocabulary_size)
        self.hidden_size = hidden_size

    def initialize_hidden_state(self, features: Tensor) -> Tensor:
        """Initialize hidden state based on encoder features."""
        return self.feature_projection(features.mean(dim=1))

    def forward(self, features: Tensor, max_len: int) -> Tensor:
        """ Generalized forward method for generating captions.

        Args:
            features: Image features of shape (batch_size, num_features, feature_size).
            max_len: Maximum length of the generated captions.

        Returns:
            Tensor: Logits of shape (batch_size, max_len, vocabulary_size).
        """
        batch_size = features.size(0)
        hidden = self.initialize_hidden_state(features)
        cell = self.initialize_cell_state(features)
        embeddings = self.embedding_layer(
            tensor(Vocabulary.sos_index,
                   device=features.device).repeat(batch_size)
        )

        logits = full(
            (batch_size, max_len, self.output_layer.out_features),
            fill_value=Vocabulary.pad_index,
            device=features.device,
            dtype=float32
        )

        for t in range(max_len):
            hidden, cell, output = self.decode_step(
                embeddings, hidden, cell, features)
            logits[:, t, :] = output

            predicted_ids = output.argmax(dim=1)
            embeddings = self.embedding_layer(predicted_ids)
            if all(predicted_ids == Vocabulary.eos_index):
                break

        return logits

    def initialize_cell_state(self, features: Tensor) -> Tensor:
        """Initialize cell state based on encoder features."""
        raise NotImplementedError(
            "This method should be overridden by subclasses.")

    def decode_step(self, embeddings: Tensor, hidden: Tensor, cell: Tensor, features: Tensor):
        """ One step of decoding. To be implemented by subclasses."""
        raise NotImplementedError(
            "This method should be overridden by subclasses.")


class GRUDecoder(CaptionDecoder):
    """GRU-based language model for generating image captions."""

    def __init__(self, vocabulary_size: int, feature_size: int, embedding_size: int = 256, hidden_size: int = 512):
        super().__init__(vocabulary_size, feature_size, embedding_size, hidden_size)
        self.gru_cell = GRUCell(embedding_size, hidden_size)

    def initialize_cell_state(self, features: Tensor) -> Tensor:
        return None

    def decode_step(self, embeddings: Tensor, hidden: Tensor, cell: Tensor, features: Tensor):
        hidden = self.gru_cell(embeddings, hidden)
        output = self.output_layer(hidden)
        return hidden, cell, output


class LSTMDecoder(CaptionDecoder):
    """LSTM-based language model for generating image captions."""

    def __init__(self, vocabulary_size: int, feature_size: int, embedding_size: int = 256, hidden_size: int = 512):
        super().__init__(vocabulary_size, feature_size, embedding_size, hidden_size)
        self.lstm_cell = LSTMCell(embedding_size, hidden_size)

    def initialize_cell_state(self, features: Tensor) -> Tensor:
        return self.feature_projection(features.mean(dim=1))

    def decode_step(self, embeddings: Tensor, hidden: Tensor, cell: Tensor, features: Tensor):
        hidden, cell = self.lstm_cell(embeddings, (hidden, cell))
        output = self.output_layer(hidden)
        return hidden, cell, output
