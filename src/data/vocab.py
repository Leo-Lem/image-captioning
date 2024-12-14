from logging import debug as DEBUG
from collections import Counter
from os import path
from pandas import DataFrame, read_csv
from spacy import blank

from __param__ import PATHS, DATA, FLAGS


def vocabularize(data: DataFrame) -> DataFrame:
    """ Extract vocabulary from the dataset. """
    if is_created():
        return load_vocab()
    text = extract_text(data)
    most_common = find_most_common(text)
    vocab = create_vocab(most_common, DATA.VOCAB_SIZE, DATA.VOCAB_THRESHOLD)
    save_vocab(vocab)
    return vocab


def is_created() -> bool:
    """ Check if the vocabulary exists. """
    return not DATA.RELOAD and path.exists(PATHS.VOCAB)


def extract_text(data: DataFrame) -> list[str]:
    """ Extract text from the dataset. """
    return [str(caption) for col in [f"caption_{i}" for i in range(1, 6)] for caption in data[col]]


def find_most_common(text: list[str]) -> list[tuple[str, int]]:
    """ Find the most common words in the text. """
    nlp, counter = blank("en"), Counter()
    for doc in nlp.pipe(text, disable=["parser", "ner"]):
        for token in doc:
            if token.is_alpha:
                counter[token.lower_] += 1
    return counter.most_common()


def create_vocab(most_common: Counter, size: int, threshold: int) -> DataFrame:
    """ Create a vocabulary from given text. """
    vocab_dict = {"<pad>": 0}
    for word, count in most_common:
        if len(vocab_dict) >= size:
            break
        if count >= threshold:
            vocab_dict[word] = len(vocab_dict)
    vocab = DataFrame(vocab_dict.items(), columns=[
                      "word", "index"]).set_index("index")
    DEBUG(f"Created vocabulary ({vocab.shape})\n{vocab.head(3)}")
    return vocab


def save_vocab(vocab: DataFrame) -> None:
    """ Save the vocabulary to a file. """
    vocab.to_csv(PATHS.VOCAB, index=True)
    DEBUG("Saved vocabulary…")


def load_vocab() -> DataFrame:
    """ Load the vocabulary from a file. """
    vocab = read_csv(PATHS.VOCAB, index_col=0)
    DEBUG(f"Loaded vocabulary ({vocab.shape})\n{vocab.head(3)}")
    return vocab
