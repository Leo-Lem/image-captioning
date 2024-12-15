from collections import Counter
from os.path import exists
from pandas import DataFrame, read_csv
from spacy import blank

from __param__ import DEBUG, PATHS, DATA, VOCAB

vocab_file = PATHS.OUT("vocab.csv")


def vocabularize(data: DataFrame) -> dict[str, int]:
    """ Extract vocabulary from the dataset. """
    if is_created():
        vocab = load_vocab()
    else:
        text = extract_text(data)
        most_common = find_most_common(text)
        vocab = create_vocab(most_common, VOCAB.SIZE, VOCAB.THRESHOLD)
        save_vocab(vocab)
    VOCAB.SIZE = len(vocab)
    return vocab


def devocabularize(vocab: dict[str, int]) -> dict[int, str]:
    """ Reverse the vocabulary. """
    reverse = {index: word for word, index in vocab.items()}
    DEBUG(f"Reversed vocabulary ({len(reverse)}")
    return reverse


def is_created() -> bool:
    """ Check if the vocabulary exists. """
    return not DATA.RELOAD and exists(vocab_file)


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
    vocab = {"<pad>": VOCAB.PADDING,
             "<start>": VOCAB.START,
             "<end>": VOCAB.END,
             "<unk>": VOCAB.UNKNOWN}
    for word, count in most_common:
        if len(vocab) >= size:
            break
        if count >= threshold:
            vocab[word] = len(vocab)
    DEBUG(f"Created vocabulary ({len(vocab)})")
    return vocab


def save_vocab(vocab: dict[str, int]):
    """ Save the vocabulary to a file. """
    DataFrame(vocab.items(), columns=["word", "index"])\
        .to_csv(vocab_file, index=False)
    DEBUG("Saved vocabulary…")


def load_vocab() -> dict[str, int]:
    """ Load the vocabulary from a file. """
    vocab = read_csv(vocab_file)
    vocab = dict(zip(vocab["word"], vocab["index"]))
    DEBUG(f"Loaded vocabulary ({len(vocab)})")
    return vocab
