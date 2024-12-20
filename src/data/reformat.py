from os.path import exists
from pandas import read_csv, DataFrame, Series

from __param__ import DEBUG, PATHS, DATA


def reformat():
    """ Reformat the Flickr8k dataset. """
    if is_reformatted():
        return

    data = load_flickr8k()
    data = group_captions(data)
    train, val, test = split(data, .8, .1, .1)
    sample = data.sample(16)
    [save(data, name)for data, name in zip([data, train, val, test, sample],
                                           ["full", "train", "val", "test", "sample"])]


def is_reformatted() -> bool:
    """ Check if the dataset has been preprocessed. """
    return not DATA.RELOAD and all([exists(PATHS.OUT(f"data-{name}.csv"))
                                    for name in ["train", "val", "test"]])


def load_flickr8k() -> DataFrame:
    """ Load the Flickr8k dataset. """
    data = read_csv(PATHS.RESOURCES("captions.csv"))
    return data


def group_captions(captions: DataFrame) -> DataFrame:
    """ Group captions by image. """
    data = captions\
        .groupby('image')['caption']\
        .apply(lambda group: list(set(group))[:5] + [""] * (5 - len(set(group))))\
        .apply(Series)\
        .rename(columns=lambda i: f"caption_{i + 1}")\
        .reset_index()
    return data


def split(data: DataFrame, train: float, val: float, test: float) -> tuple[DataFrame, DataFrame, DataFrame]:
    """ Split the dataset into train, validation, and test sets. """

    data = data.sample(frac=1)
    train_end = int(train * len(data))
    val_end = train_end + int(val * len(data))
    train, val, test = data[:train_end], data[train_end:val_end], data[val_end:]
    return train, val, test


def save(data: DataFrame, name: str):
    """ Save the preprocessed dataset. """
    data.to_csv(PATHS.OUT(f"data-{name}.csv"), index=False)
