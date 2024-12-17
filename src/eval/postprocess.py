from torch import Tensor
from torch.nn.functional import softmax

from __param__ import DATA, DEBUG
from src.data import Vocabulary


class CaptionPostprocessor:
    """ Postprocess the captions """

    def __init__(self):
        self.vocab = Vocabulary()

    def __call__(self, indexed: Tensor) -> list[str]:
        """ Convert the indices to strings. """
        assert indexed.size(1) <= DATA.CAPTION_LEN
        caption = [self.stringify(self.retokenize(indices))
                   for indices in indexed]
        return caption

    def retokenize(self, indices: Tensor) -> list[str]:
        """ Convert a tokenized caption to a string. """
        assert indices.size(0) <= DATA.CAPTION_LEN
        tokenized = filter(lambda x: x != "<unknown>",
                           [self.vocab[index] for index in indices.tolist()
                            if index not in (Vocabulary.PADDING, Vocabulary.UNKNOWN, Vocabulary.START, Vocabulary.END)])
        return tokenized

    def stringify(self, tokenized: list[str]) -> str:
        """ Convert a tokenized caption to a string. """
        return " ".join(tokenized)
