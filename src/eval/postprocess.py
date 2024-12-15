from torch import Tensor
from torch.nn.functional import softmax

from __param__ import DATA, DEBUG
from src.data import Vocabulary


class CaptionPostprocessor:
    """ Postprocess the captions """

    def __init__(self):
        self.vocab = Vocabulary()

    def __call__(self, embeddings: Tensor) -> list[str]:
        """ Extract the most probable token from the embeddings. """
        batch_size = embeddings.size(0)
        assert embeddings.size() == (batch_size, DATA.CAPTION_LEN, Vocabulary.SIZE)

        indexed = embeddings.argmax(dim=-1)
        assert indexed.size() == (batch_size, DATA.CAPTION_LEN)

        captions = [self.stringify(self.retokenize(indices))
                    for indices in indexed]
        return captions

    def extract_from_indexed(self, indexed: Tensor) -> list[list[str]]:
        """ Extract strings from tokenized captions. """
        assert indexed.size() == (indexed.size(0), DATA.NUM_CAPTIONS, DATA.CAPTION_LEN)
        tokenizeds = [[self.stringify(self.retokenize(caption))
                       for caption in batch]
                      for batch in indexed]
        return tokenizeds

    def retokenize(self, indexed: Tensor) -> list[str]:
        """ Convert a tokenized caption to a string. """
        assert indexed.size() == (DATA.CAPTION_LEN,)
        tokenized = filter(lambda x: x != "<unknown>",
                           [self.vocab[index] for index in indexed.tolist()
                            if index not in (Vocabulary.PADDING, Vocabulary.UNKNOWN, Vocabulary.START, Vocabulary.END)])
        return tokenized

    def stringify(self, tokenized: list[str]) -> str:
        """ Convert a tokenized caption to a string. """
        return " ".join(tokenized)
