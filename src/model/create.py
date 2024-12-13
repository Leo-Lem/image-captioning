from .decoder import Decoder, GRUDecoder, LSTMDecoder, TransformerDecoder
from __param__ import APPROACH


def create_model() -> Decoder:
    """
    Create a new decoder model based on the specified approach.

    :return: The initialized model.
    :raises ValueError: If an unknown approach is specified.
    """
    embedding_dim = 256
    hidden_dim = 512
    vocab_size = 10000
    num_layers = 2
    dropout = 0.1

    if APPROACH == "gru":
        return GRUDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers, dropout=dropout)
    elif APPROACH == "lstm":
        return LSTMDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=num_layers, dropout=dropout)
    elif APPROACH == "transformer":
        num_heads = 8  # Example number of attention heads for Transformer
        return TransformerDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown approach: {APPROACH}")
