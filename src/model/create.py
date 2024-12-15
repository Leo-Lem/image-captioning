from __param__ import MODEL
from .decoder import Decoder
from .gru import GRUDecoder
from .lstm import LSTMDecoder
from .transformer import TransformerDecoder


def decoder() -> Decoder:
    """ Create a new decoder model based on the specified approach. """
    if "gru" in MODEL.NAME:
        return GRUDecoder()
    elif "lstm" in MODEL.NAME:
        return LSTMDecoder()
    elif "transformer" in MODEL.NAME:
        return TransformerDecoder()
    else:
        raise ValueError(f"Unknown model: {MODEL.NAME}")
