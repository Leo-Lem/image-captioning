from logging import debug as DEBUG
from collections import Counter
from os import path
from pandas import DataFrame, read_csv
from spacy import blank

from __param__ import PATHS, DATA, FLAGS


def vocabularize(data: DataFrame) -> dict[str, int]:
    """ Extract vocabulary from the dataset. """
    if is_created():
        vocab = load_vocab()
    else:
        text = extract_text(data)
        most_common = find_most_common(text)
        vocab = create_vocab(most_common,
                             DATA.VOCAB_SIZE,
                             DATA.VOCAB_THRESHOLD)
        save_vocab(vocab)
    DATA.VOCAB_SIZE = len(vocab)
    return vocab


def devocabularize(vocab: dict[str, int]) -> dict[int, str]:
    """ Reverse the vocabulary. """
    reverse = {index: word for word, index in vocab.items()}
    DEBUG(f"Reversed vocabulary ({len(reverse)})\n{reverse}")
    return reverse


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


def create_vocab(most_common: Counter, size: int, threshold: int) -> dict[str, int]:
    """ Create a vocabulary from given text. """
    vocab = {"<pad>": DATA.PADDING,
             "<start>": DATA.START,
             "<end>": DATA.END,
             "<unk>": DATA.UNKNOWN}
    for word, count in most_common:
        if len(vocab) >= size:
            break
        if count >= threshold:
            vocab[word] = len(vocab)
    DEBUG(f"Created vocabulary ({len(vocab)})\n{vocab}")
    return vocab


def save_vocab(vocab: dict[str, int]):
    """ Save the vocabulary to a file. """
    vocab = DataFrame(vocab.items(), columns=["word", "index"])
    vocab.to_csv(PATHS.VOCAB, index=False)
    DEBUG("Saved vocabulary…")


def load_vocab() -> dict[str, int]:
    """ Load the vocabulary from a file. """
    vocab = read_csv(PATHS.VOCAB)
    vocab = dict(zip(vocab["word"], vocab["index"]))
    DEBUG(f"Loaded vocabulary ({len(vocab)})\n{vocab}")
    return vocab