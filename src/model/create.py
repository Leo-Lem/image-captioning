from __param__ import MODEL
from .decoder import Decoder
from .gru import GRUDecoder
from .lstm import LSTMDecoder
from .transformer import TransformerDecoder


def decoder() -> Decoder:
    """ Create a new decoder model based on the specified approach. """
    if MODEL.APPROACH == "gru":
        return GRUDecoder()
    elif MODEL.APPROACH == "lstm":
        return LSTMDecoder()
    elif MODEL.APPROACH == "transformer":
        return TransformerDecoder()
    else:
        raise ValueError(f"Unknown approach: {MODEL.APPROACH}")
