from collections import Counter
from importlib import import_module
from spacy import Language
from spacy.cli import download
from tqdm import tqdm


class Vocabulary:
    """ Vocabulary object to store the mapping between words and indices. """

    pad_token = "<PAD>"
    sos_token = "<SOS>"
    eos_token = "<EOS>"
    unk_token = "<UNK>"
    pad_index = 0
    sos_index = 1
    eos_index = 2
    unk_index = 3

    def __init__(self, sentences: list[str], threshold: int, spacy_model_name: str = "en_core_web_sm"):
        """ Initialize the vocabulary and learn the words from the sentences.

        Args:
            threshold: The minimum frequency for a word to be included in the vocabulary.
            sentences: A list of sentences to learn the vocabulary from.
            spacy_model_name: The name of the spaCy model to use for tokenization.
        """
        self.index_to_token = {self.pad_index: self.pad_token,
                               self.sos_index: self.sos_token,
                               self.eos_index: self.eos_token,
                               self.unk_index: self.unk_token}
        self.threshold = threshold
        self.spacy_model = self._download_and_init(spacy_model_name)

        freq = Counter()

        # Use tqdm to provide a progress bar for vocabulary creation
        for sentence in tqdm(sentences, desc="Building Vocabulary", unit="sentences"):
            for token in self.tokenize(sentence):
                freq[token] += 1

                if token not in self.index_to_token.values():
                    self.index_to_token[len(self.index_to_token)] = token

    def __len__(self) -> int:
        return len(self.index_to_token)

    @property
    def token_to_index(self) -> dict[str, int]:
        return {v: k for k, v in self.index_to_token.items()}

    def numericalize(self, text: str) -> list[int]:
        """ Convert the text to numerical form using the vocabulary. """
        return [self.token_to_index[token]
                if token in self.token_to_index else self.token_to_index[self.unk_token]
                for token in self.tokenize(text)]

    def denumericalize(self, indices: list[int]) -> str:
        """ Convert the indices back to text using the vocabulary. """
        return " ".join([self.index_to_token[index] for index in indices])

    def tokenize(self, text: str) -> list[str]:
        """ Tokenize the text using spacy model. """
        return [token.text.lower() for token in self.spacy_model.tokenizer(text)]

    def _download_and_init(self, model_name: str) -> Language:
        """ Load a spaCy model, download it if it has not been installed yet.

        Args:
            model_name: The name of the spaCy model to load.
        """
        try:
            model_module = import_module(model_name)
        except ModuleNotFoundError:
            download(model_name)
            model_module = import_module(model_name)

        return model_module.load()
