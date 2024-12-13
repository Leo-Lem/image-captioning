from ast import literal_eval
from pandas import DataFrame, read_csv
from torch import tensor, stack, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from __param__ import PATHS, TRAIN


def preprocess(dataset: str) -> DataLoader:
    """
    Load, preprocess, and return a DataLoader for the specified dataset.

    Args:
        dataset (str): The dataset to load ("train", "val", "test", "sample").
        vocab (dict): Vocabulary mapping words to indices.
        max_caption_length (int): Maximum length for captions (default=20).

    Returns:
        DataLoader: Preprocessed DataLoader for the dataset.
    """
    data: DataFrame = load(dataset)
    vocab: DataFrame = read_csv(f"{PATHS.RES}/vocab.csv").to_dict()
    max_caption_length: int = 20

    # Parse vectors and preprocess captions
    parsed_data: list = []
    for _, row in data.iterrows():
        # Parse the image vector
        vector: Tensor = tensor(literal_eval(row["vector"])).float()

        # Parse and tokenize one caption from the caption set
        captions: set = literal_eval(row["captions"])
        # Pick the first caption for simplicity
        selected_caption: str = list(captions)[0]
        tokenized_caption: list = tokenize_caption(
            selected_caption, vocab, max_caption_length)

        # Append processed data
        parsed_data.append((vector, tokenized_caption))

    # Define a Dataset class inline
    class ProcessedDataset(Dataset):
        def __init__(self, data: list) -> None:
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple:
            vector, tokenized_caption = self.data[idx]
            # Exclude last token for inputs
            inputs: Tensor = tensor(tokenized_caption[:-1])
            # Exclude first token for targets
            targets: Tensor = tensor(tokenized_caption[1:])
            return vector, inputs, targets

    # Create a DataLoader
    is_train: bool = dataset == "train"
    dataloader: DataLoader = DataLoader(
        ProcessedDataset(parsed_data),
        batch_size=TRAIN.BATCH_SIZE,
        shuffle=is_train,
        collate_fn=lambda batch: (
            stack([item[0] for item in batch]),  # Image features
            pad_sequence([item[1] for item in batch], batch_first=True,
                         padding_value=vocab["<pad>"]),  # Inputs
            pad_sequence([item[2] for item in batch], batch_first=True,
                         padding_value=vocab["<pad>"])   # Targets
        ),
    )

    return dataloader


def load(dataset: str) -> DataFrame:
    """
    Load the specified dataset.

    Args:
        dataset (str): The dataset to load ("train", "val", "test", "sample").

    Returns:
        DataFrame: Loaded dataset.
    """
    assert dataset in ["train", "val", "test", "sample"], "Invalid dataset"
    return read_csv(f"{PATHS.RES}/{dataset}.csv")


def tokenize_caption(caption: str, vocab: dict, max_caption_length: int) -> list[int]:
    """
    Tokenize a caption into indices based on the vocabulary.

    Args:
        caption (str): Input caption.
        vocab (dict): Vocabulary mapping words to indices.
        max_caption_length (int): Maximum caption length for padding/truncation.

    Returns:
        list: Tokenized caption as a list of indices.
    """
    tokens = caption.lower().split()
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    # Pad or truncate to max_caption_length
    indices = indices[:max_caption_length] + [vocab["<pad>"]]\
        * max(0, max_caption_length - len(indices))
    return indices
