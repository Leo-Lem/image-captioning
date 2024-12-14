from __param__ import APPROACH
from .decoder import Decoder
from .gru import GRUDecoder
from .lstm import LSTMDecoder
from .transformer import TransformerDecoder


def create_decoder() -> Decoder:
    """ Create a new decoder model based on the specified approach. """
    if APPROACH == "gru":
        return GRUDecoder()
    elif APPROACH == "lstm":
        return LSTMDecoder()
    elif APPROACH == "transformer":
        return TransformerDecoder()
    else:
        raise ValueError(f"Unknown approach: {APPROACH}")
