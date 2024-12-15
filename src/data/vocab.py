from collections import Counter
from os.path import exists
from pandas import DataFrame, read_csv
from spacy import blank

from __param__ import DEBUG, PATHS, DATA


class Vocabulary:
    """ Vocabulary for the image captioning dataset. """

    DATA, FILE = PATHS.OUT("data-full.csv"), PATHS.OUT("vocab.csv")
    SIZE, THRESHOLD = 8096, 3
    PADDING, UNKNOWN, START, END = 0, 1, 2, 3

    def __init__(self):
        assert exists(self.DATA), f"Data file not found: {self.DATA}"
        if self._is_created():
            self._load()
        else:
            self.data = read_csv(self.DATA)
            self._token_to_index = {"<pad>": self.PADDING,
                                    "<unknown>": self.UNKNOWN,
                                    "<start>": self.START,
                                    "<end>": self.END}
            text = self._text()
            most_common = self._most_common(text)
            self._assemble(most_common)
            self._index_to_token = {index: word
                                    for word, index in self._token_to_index.items()}
            self._save()

    def _is_created(self) -> bool:
        """ Check if the vocabulary exists. """
        return not DATA.RELOAD and exists(self.FILE)

    def _save(self):
        DataFrame(self._token_to_index.items(), columns=["word", "index"])\
            .to_csv(self.FILE, index=False)
        DEBUG("Saved vocabulary…")

    def _load(self):
        vocab = read_csv(self.FILE)
        self._token_to_index = dict(zip(vocab["word"], vocab["index"]))
        self._index_to_token = {index: word for word,
                                index in self._token_to_index.items()}

    def _text(self) -> list[str]:
        """ Extract text from the dataset. """
        return [str(caption)
                for col in [f"caption_{i}" for i in range(1, 6)]
                for caption in self.data[col]]

    def _most_common(self, text: list[str]) -> list[tuple[str, int]]:
        """ Find the most common words in the text. """
        nlp, counter = blank("en"), Counter()
        for doc in nlp.pipe(text, disable=["parser", "ner"]):
            for token in doc:
                if token.is_alpha:
                    counter[token.lower_] += 1
        return counter.most_common()

    def _assemble(self, most_common: Counter):
        """ Create a vocabulary from given word counter. """
        for word, count in most_common:
            if len(self) >= self.SIZE:
                break
            if count >= self.THRESHOLD:
                self._token_to_index[word] = len(self)

    def __len__(self):
        return len(self._token_to_index)

    def __getitem__(self, key: str | int) -> int | str:
        if isinstance(key, str):
            return self._token_to_index.get(key, self.UNKNOWN)
        if isinstance(key, int):
            return self._index_to_token.get(key, "<unknown>")
        raise KeyError(f"Invalid key type: {type(key)}")

    def __contains__(self, key: str | int) -> bool:
        if isinstance(key, str):
            return key in self._token_to_index
        if isinstance(key, int):
            return key in self._index_to_token
        return False

    def __iter__(self) -> iter:
        """ Iterate over the vocabulary (token, index) """
        return iter(self._token_to_index)

    def __str__(self) -> str:
        return f"Vocabulary ({len(self)} words)"
