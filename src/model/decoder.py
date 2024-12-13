from torch import Tensor, cat, zeros
from torch.nn import Module, Embedding, Dropout, LSTM, Linear, GRU, Transformer, Parameter


class Decoder(Module):
    """ Base class for decoders with common functionality for embedding and dropout. """

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, dropout: float = 0.1) -> None:
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.dropout = Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, image_features: Tensor, captions: Tensor) -> Tensor:
        """
        Forward pass for the decoder. Should be implemented by subclasses.

        Args:
            image_features (Tensor): Image feature vectors, shape (batch_size, feature_dim).
            captions (Tensor): Input caption tokens, shape (batch_size, seq_len).

        Returns:
            Tensor: Predicted token probabilities, shape (batch_size, seq_len, vocab_size).
        """
        raise NotImplementedError("Subclasses should implement this!")


class GRUDecoder(Decoder):
    """ GRU-based decoder for sequence generation. """

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, num_layers: int = 1, dropout: float = 0.1) -> None:
        super(GRUDecoder, self).__init__(
            embedding_dim, hidden_dim, vocab_size, dropout)
        self.gru = GRU(embedding_dim, hidden_dim, num_layers,
                       batch_first=True, dropout=dropout)
        self.fc = Linear(hidden_dim, vocab_size)

    def forward(self, image_features: Tensor, captions: Tensor) -> Tensor:
        """ Forward pass for the GRU decoder. """
        # (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(captions)
        # Include image features
        embeddings = cat((image_features.unsqueeze(1), embeddings), dim=1)
        # (batch_size, seq_len+1, hidden_dim)
        outputs, _ = self.gru(embeddings)
        # (batch_size, seq_len+1, vocab_size)
        outputs = self.fc(outputs)
        return outputs


class LSTMDecoder(Decoder):
    """ LSTM-based decoder for sequence generation. """

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, num_layers: int = 1, dropout: float = 0.1) -> None:
        super(LSTMDecoder, self).__init__(
            embedding_dim, hidden_dim, vocab_size, dropout)
        self.lstm = LSTM(embedding_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=dropout)
        self.fc = Linear(hidden_dim, vocab_size)

    def forward(self, image_features: Tensor, captions: Tensor) -> Tensor:
        """ Forward pass for the LSTM decoder. """
        embeddings = self.embedding(
            captions)  # (batch_size, seq_len, embedding_dim)
        # Include image features
        embeddings = cat((image_features.unsqueeze(1), embeddings), dim=1)
        # (batch_size, seq_len+1, hidden_dim)
        outputs, _ = self.lstm(embeddings)
        outputs = self.fc(outputs)  # (batch_size, seq_len+1, vocab_size)
        return outputs


class TransformerDecoder(Decoder):
    """ Transformer-based decoder for sequence generation. """

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, num_heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super(TransformerDecoder, self).__init__(
            embedding_dim, hidden_dim, vocab_size, dropout)
        self.positional_encoding = Parameter(
            zeros(1, 100, embedding_dim))  # Max length = 100
        self.transformer = Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=0,  # We only need a decoder
            num_decoder_layers=num_layers,
            dropout=dropout,
        )
        self.fc = Linear(embedding_dim, vocab_size)

    def forward(self, image_features: Tensor, captions: Tensor) -> Tensor:
        """ Forward pass for the Transformer decoder. """
        # (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(captions)
        # Add positional encoding
        embeddings += self.positional_encoding[:, :captions.size(1), :]
        # Repeat features
        image_features = image_features\
            .unsqueeze(1)\
            .repeat(1, captions.size(1), 1)
        # Combine features and embeddings
        embeddings = cat((image_features, embeddings), dim=1)
        tgt_mask = Transformer\
            .generate_square_subsequent_mask(embeddings.size(1))\
            .to(embeddings.device)
        # (seq_len, batch_size, d_model)
        outputs = self.transformer(embeddings
                                   .permute(1, 0, 2), tgt_mask=tgt_mask)
        # (batch_size, seq_len, vocab_size)
        outputs = self.fc(outputs.permute(1, 0, 2))
        return outputs
