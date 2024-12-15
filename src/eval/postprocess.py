from torch import Tensor
from torch.nn.functional import softmax

from __param__ import DATA, VOCAB


def extract_from_embedding(embeddings: Tensor, reversed_vocab: dict[int, str]) -> list[str]:
    """ Extract the most probable token from the embeddings. """
    batch_size = embeddings.size(0)
    assert embeddings.size() == (batch_size, DATA.CAPTION_LEN, VOCAB.SIZE)

    probabilities = softmax(embeddings, dim=-1)
    assert probabilities.size() == (batch_size, DATA.CAPTION_LEN, VOCAB.SIZE)

    tokenized = probabilities.argmax(dim=-1)
    assert tokenized.size() == (batch_size, DATA.CAPTION_LEN)

    captions = [stringify(caption, reversed_vocab)
                for caption in tokenized]
    return captions


def extract_from_tokenized(tokenized: Tensor, reversed_vocab: dict[int, str]) -> list[list[str]]:
    """ Extract strings from tokenized captions. """
    assert tokenized.size() == (tokenized.size(0), DATA.NUM_CAPTIONS, DATA.CAPTION_LEN)
    captions = [[stringify(caption, reversed_vocab)
                for caption in batch]
                for batch in tokenized]
    return captions


def stringify(tokenized: Tensor, reversed_vocab: dict[int, str]) -> str:
    """ Convert a tokenized caption to a string. """
    assert tokenized.size() == (DATA.CAPTION_LEN,), tokenized.size()
    caption = [reversed_vocab[token] for token in tokenized.tolist()
               if token not in (VOCAB.START, VOCAB.PADDING, VOCAB.END, VOCAB.UNKNOWN)]
    return " ".join(caption)
